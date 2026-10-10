#!/usr/bin/env python3
"""
方案B — #8 JHTDB 扩样评估 (38 original + 80 expanded = 118 samples)
===================================================================
Evaluate FNO V3/V4 zero-shot on the COMBINED JHTDB validation set, with:
  - pooled rel-L2, MSE, MAE, correlation
  - per-sample rel-L2 -> mean+-std, bootstrap 95% CI
  - paired Wilcoxon + paired t-test V3 vs V4

Output: results/jhtdb_expanded_stats.json
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F

BASE="/mnt/pfs/navier-stokes"
sys.path.insert(0, BASE)
from train_fno_v3 import VorticityFNO3D
import scipy.stats

torch.manual_seed(0); np.random.seed(0)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, flush=True)

def load(ck, device):
    m=VorticityFNO3D(in_c=3,out_c=3,width=64,modes_h=12,modes_w=12,modes_d=8,n_layers=6).to(device)
    c=torch.load(ck, map_location=device)
    sd=c.get('model_state_dict',c); m.load_state_dict(sd); m.eval()
    return m

@torch.no_grad()
def predict(model, omega, stats, bs=32):
    out=[]
    im,isv,tm,ts=stats['in_mean'],stats['in_std'],stats['tg_mean'],stats['tg_std']
    for i in range(0,omega.shape[0],bs):
        b=omega[i:i+bs].to(DEVICE)
        p=model((b-im)/isv); out.append((p*ts+tm).cpu())
    return torch.cat(out,0)

def main():
    with open(os.path.join(BASE,"results","jhtdb_validation_v2_results.json")) as f:
        ns=json.load(f)["normalization_stats"]
    stats={'in_mean':ns['in_mean'],'in_std':ns['in_std'],'tg_mean':ns['tg_mean'],'tg_std':ns['tg_std']}

    # load original 38
    d1=np.load(os.path.join(BASE,"jhtdb_data","jhtdb_validation_data.npz"))
    o1=torch.from_numpy(d1["omega_input"]).float(); t1=torch.from_numpy(d1["omega_target"]).float()
    # load expanded 80
    d2=np.load(os.path.join(BASE,"jhtdb_data","jhtdb_validation_expanded.npz"))
    o2=torch.from_numpy(d2["omega_input"]).float(); t2=torch.from_numpy(d2["omega_target"]).float()
    om_in=torch.cat([o1,o2],dim=0); om_tg=torch.cat([t1,t2],dim=0)
    print(f"Combined JHTDB: {om_in.shape} (38 + {o2.shape[0]})", flush=True)
    print(f"  input range [{om_in.min():.4f},{om_in.max():.4f}] target [{om_tg.min():.4f},{om_tg.max():.4f}]", flush=True)

    ckpts={'v3':os.path.join(BASE,"checkpoints/fno_v3_best.pt"),
           'v4':os.path.join(BASE,"checkpoints/fno_v4_best.pt")}
    results={'n':int(om_in.shape[0]),'n_orig':38,'n_expanded':int(o2.shape[0])}
    per={}
    for name,path in ckpts.items():
        if not os.path.exists(path): print("missing",path); continue
        model=load(path,DEVICE)
        pred=predict(model, om_in, stats)
        pf=pred.flatten(1); tf=om_tg.flatten(1)
        per_rel=(pf.norm(dim=1)/(tf.norm(dim=1)+1e-8)).numpy()
        per_mse=((pred-om_tg)**2).flatten(1).mean(1).numpy()
        pc=pf-pf.mean(1,keepdim=True); tc=tf-tf.mean(1,keepdim=True)
        per_corr=(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)+1e-8)
        per_corr=per_corr.numpy()
        # pooled (global) rel-L2
        pooled=(pred-om_tg).norm().item()/(om_tg.norm().item()+1e-8)
        # bootstrap CI on per-sample rel-L2 mean
        rng=np.random.RandomState(0); n=per_rel.shape[0]; B=10000
        bmeans=np.array([per_rel[rng.randint(0,n,n)].mean() for _ in range(B)])
        results[name]={'pooled_rel_l2':float(pooled),
            'per_sample_rel_l2_mean':float(per_rel.mean()),'std':float(per_rel.std(ddof=1)),
            'ci95':[float(np.percentile(bmeans,2.5)),float(np.percentile(bmeans,97.5))],
            'mse':F.mse_loss(pred,om_tg).item(),'mae':F.l1_loss(pred,om_tg).item(),
            'corr':float(per_corr.mean())}
        per[name]=per_rel
        print(f"{name}: pooled_rel_L2={pooled:.4f} per_sample={per_rel.mean():.4f}+-{per_rel.std(ddof=1):.4f} CI95=[{np.percentile(bmeans,2.5):.4f},{np.percentile(bmeans,97.5):.4f}]", flush=True)
        del model,pred; gc.collect(); torch.cuda.empty_cache()
    if 'v3' in per and 'v4' in per:
        diff=per['v4']-per['v3']
        t,p_t=scipy.stats.ttest_rel(per['v4'],per['v3'])
        w,p_w=scipy.stats.wilcoxon(per['v4'],per['v3'])
        rng=np.random.RandomState(1)
        bd=np.array([diff[rng.randint(0,len(diff),len(diff))].mean() for _ in range(10000)])
        results['paired_v3_vs_v4']={'mean_diff_v4_v3':float(diff.mean()),
            'ci95':[float(np.percentile(bd,2.5)),float(np.percentile(bd,97.5))],
            'ttest_p':float(p_t),'wilcoxon_p':float(p_w)}
        print(f"Paired V4-V3 diff={diff.mean():.4f} CI=[{np.percentile(bd,2.5):.4f},{np.percentile(bd,97.5):.4f}] wilcoxon_p={p_w:.2e} ttest_p={p_t:.2e}", flush=True)
    out=os.path.join(BASE,"results","jhtdb_expanded_stats.json")
    with open(out,'w') as f: json.dump(results,f,indent=1)
    print("Saved",out,flush=True)

if __name__=='__main__':
    main()
