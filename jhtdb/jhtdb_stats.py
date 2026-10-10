#!/usr/bin/env python3
"""
方案B — #8 JHTDB 统计稳健性分析 (38 samples)
============================================
Directly addresses the reviewer's "sample size too small" concern by quantifying
the uncertainty of the zero-shot JHTDB evaluation:
  - per-sample relative-L2 errors for V3 and V4
  - mean +- std, 95% CI via bootstrap (10k resamples)
  - paired Wilcoxon signed-rank test on V3 vs V4 per-sample rel-L2
  - report whether the V3-beats-V4 difference is statistically significant
  - also compute per-sample correlation and MSE for completeness

Output: results/jhtdb_stats_results.json
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F

BASE="/mnt/pfs/navier-stokes"
sys.path.insert(0, BASE)
from train_fno_v3 import VorticityFNO3D

torch.manual_seed(0); np.random.seed(0)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, flush=True)
import scipy.stats

def load(ck, device):
    m=VorticityFNO3D(in_c=3,out_c=3,width=64,modes_h=12,modes_w=12,modes_d=8,n_layers=6).to(device)
    c=torch.load(ck, map_location=device)
    sd=c.get('model_state_dict',c); m.load_state_dict(sd); m.eval()
    return m

def predict(model, omega, stats, bs=32):
    out=[]
    im,isv,tm,ts=stats['in_mean'],stats['in_std'],stats['tg_mean'],stats['tg_std']
    with torch.no_grad():
        for i in range(0,omega.shape[0],bs):
            b=omega[i:i+bs].to(DEVICE)
            p=model((b-im)/isv); out.append((p*ts+tm).cpu())
    return torch.cat(out,0)

def main():
    # Load normalization stats from the SAVED validation v2 result (avoids slow regeneration)
    with open(os.path.join(BASE,"results","jhtdb_validation_v2_results.json")) as f:
        v2=json.load(f)
    ns=v2["normalization_stats"]
    stats={'in_mean':ns['in_mean'],'in_std':ns['in_std'],'tg_mean':ns['tg_mean'],'tg_std':ns['tg_std']}
    print("stats (from saved v2):",{k:round(v,5) for k,v in stats.items()}, flush=True)

    d=np.load(os.path.join(BASE,"jhtdb_data","jhtdb_validation_data.npz"))
    om_in=torch.from_numpy(d["omega_input"]).float()
    om_tg=torch.from_numpy(d["omega_target"]).float()
    print("JHTDB:", om_in.shape, "range", float(om_in.min()), float(om_in.max()), flush=True)

    ckpts={'v3':os.path.join(BASE,"checkpoints/fno_v3_best.pt"),
           'v4':os.path.join(BASE,"checkpoints/fno_v4_best.pt")}
    results={}
    for name,path in ckpts.items():
        if not os.path.exists(path): print("missing",path); continue
        model=load(path,DEVICE)
        pred=predict(model, om_in, stats)
        # per-sample rel-L2
        pred_f=pred.flatten(1); tg_f=om_tg.flatten(1)
        per_rel = (pred_f.norm(dim=1)/(tg_f.norm(dim=1)+1e-8)).numpy()
        per_mse = ((pred-om_tg)**2).flatten(1).mean(1).numpy()
        # per-sample correlation
        pf=pred.flatten(1); tf=om_tg.flatten(1)
        pc=pf-pf.mean(1,keepdim=True); tc=tf-tf.mean(1,keepdim=True)
        per_corr=(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)+1e-8)
        per_corr=per_corr.numpy()
        mean_rel,std_rel=per_rel.mean(),per_rel.std(ddof=1)
        mean_mse,std_mse=per_mse.mean(),per_mse.std(ddof=1)
        # bootstrap 95% CI
        rng=np.random.RandomState(0); n=per_rel.shape[0]; B=10000
        bmeans=np.array([per_rel[rng.randint(0,n,n)].mean() for _ in range(B)])
        ci_lo,ci_hi=np.percentile(bmeans,[2.5,97.5])
        results[name]={'n':int(n),'mean_rel_l2':float(mean_rel),'std_rel_l2':float(std_rel),
                       'ci95_rel_l2':[float(ci_lo),float(ci_hi)],
                       'mean_mse':float(mean_mse),'std_mse':float(std_mse),
                       'mean_corr':float(per_corr.mean()),'std_corr':float(per_corr.std(ddof=1))}
        print(f"{name}: rel_L2={mean_rel:.4f}+-{std_rel:.4f} CI95=[{ci_lo:.4f},{ci_hi:.4f}] corr={per_corr.mean():.3f} n={n}", flush=True)
        del model,pred; gc.collect(); torch.cuda.empty_cache()
    # Paired test V3 vs V4 (per-sample rel-L2, same samples)
    if 'v3' in results and 'v4' in results:
        # recompute per-sample for both to pair
        per={}
        for name in['v3','v4']:
            model=load(ckpts[name],DEVICE); pred=predict(model,om_in,stats)
            pf=pred.flatten(1); tf=om_tg.flatten(1)
            per[name]=(pf.norm(dim=1)/(tf.norm(dim=1)+1e-8)).numpy()
        diff=per['v4']-per['v3']  # positive means v4 worse
        # paired t-test and Wilcoxon
        t,p_t=scipy.stats.ttest_rel(per['v4'],per['v3'])
        w,p_w=scipy.stats.wilcoxon(per['v4'],per['v3'])
        # bootstrap CI on mean difference
        rng=np.random.RandomState(1)
        bd=np.array([diff[rng.randint(0,len(diff),len(diff))].mean() for _ in range(10000)])
        results['paired_v3_vs_v4']={'mean_rel_L2_diff_(v4-v3)':float(diff.mean()),
            'ci95_diff':[float(np.percentile(bd,2.5)),float(np.percentile(bd,97.5))],
            'paired_t_pvalue':float(p_t),'wilcoxon_pvalue':float(p_w)}
        print(f"Paired V4-V3 rel_L2 diff={diff.mean():.4f} CI95=[{np.percentile(bd,2.5):.4f},{np.percentile(bd,97.5):.4f}] wilcoxon_p={p_w:.4f}",flush=True)
    out=os.path.join(BASE,"results","jhtdb_stats_results.json")
    with open(out,'w') as f: json.dump(results,f,indent=1)
    print("Saved",out,flush=True)

if __name__=='__main__':
    main()
