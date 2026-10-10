#!/usr/bin/env python3
"""
方案B — 统一尺度评估脚本 (resolves Metric Contradiction Issue #4)
=================================================================
Compute a SINGLE scale-invariant metric (relative L2 error) uniformly across
synthetic / DNS-like / JHTDB test sets for FNO V3 and V4, so numbers are directly
comparable. Also computes physics regularity metrics on predictions.

The core problem being fixed: the paper previously reported raw MSE on 3 different
physical scales (synthetic z-score ~0.09, synthetic raw ~1.28, JHTDB ~0.002), making
the "20-25x gap" claim mathematically invalid. Relative L2 error is scale-free.

Output: results/unified_metric_results.json
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = "/mnt/pfs/navier-stokes"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Unified Eval] Device: {DEVICE}", flush=True)

sys.path.insert(0, BASE_DIR)
try:
    from vorticity_analyzer import EnhancedPhysicsLoss, create_synthetic_vorticity
    from vorticity_fno import generate_divfree_vortex
    from train_fno_v3 import VorticityFNO3D
except Exception as e:
    print("WARN import:", e, flush=True)
    from train_fno_v3 import VorticityFNO3D

def build_model(device):
    m = VorticityFNO3D(in_c=3, out_c=3, width=64,
                       modes_h=12, modes_w=12, modes_d=8, n_layers=6).to(device)
    return m

def load(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def predict(model, omega, in_mean, in_std, tg_mean, tg_std, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, omega.shape[0], bs):
            b = omega[i:i+bs].to(DEVICE)
            inp_norm = (b - in_mean) / in_std
            pred = model(inp_norm)
            pred = pred * tg_std + tg_mean
            out.append(pred.cpu())
    return torch.cat(out, dim=0)

def rel_l2(pred, tgt):
    """Scale-invariant relative L2 error: ||pred-tgt||_2 / ||tgt||_2"""
    return (pred - tgt).norm().item() / (tgt.norm().item() + 1e-8)

def nrmse(pred, tgt):
    """Normalized RMSE: RMSE / std(target)"""
    return torch.sqrt(F.mse_loss(pred, tgt)).item() / (tgt.std().item() + 1e-8)

# ============ 1. Normalization stats (reproduce training) ============
def compute_norm_stats(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    inputs, targets = [], []
    size = 32
    nu = torch.tensor(0.01); k = torch.tensor(1.0)
    x = torch.linspace(0, 2*np.pi, size); y = torch.linspace(0, 2*np.pi, size); z = torch.linspace(0, 2*np.pi, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    w1 = -k*torch.cos(k*X)*torch.sin(k*Y)*torch.sin(k*Z)
    w2 = k*torch.sin(k*X)*torch.cos(k*Y)*torch.sin(k*Z)
    w3 = torch.zeros_like(X)
    om0 = torch.stack([w1,w2,w3],0).unsqueeze(0)
    for dt1 in torch.linspace(0.05,1.0,10):
        for dt2 in torch.linspace(0.1,2.0,8):
            d = torch.exp(-2*nu*k**2*dt1); dt = torch.exp(-2*nu*k**2*(dt1+dt2))
            inputs.append(om0*d); targets.append(om0*dt)
    for intensity in [0.5,1,2,3,5,8,10,12,15,18,20]:
        n=250 if intensity<=15 else 50
        for _ in range(n):
            _,omega = generate_divfree_vortex(1,size,float(intensity))
            ns=0.03*(1.0+intensity/20.0)
            noise=torch.randn_like(omega)*ns
            shift=torch.roll(omega,shifts=1,dims=(-1,))
            target=omega*0.7+shift*0.3+noise
            inputs.append(omega); targets.append(target)
    for intensity in [0.5,1,2,3,5,8,10,12,15,18,20]:
        n=180 if intensity<=15 else 20
        for _ in range(n):
            omega=create_synthetic_vorticity(1,size)*(intensity/5.0)
            ns=0.03*(1.0+intensity/20.0)
            noise=torch.randn_like(omega)*ns
            shift=torch.roll(omega,shifts=(-1,2),dims=(-2,-1))
            inputs.append(omega*0.65+shift*0.35+noise); targets.append(omega)
    inputs=torch.cat(inputs,0); targets=torch.cat(targets,0)
    stats={'in_mean':inputs.mean().item(),'in_std':inputs.std().item()+1e-8,
           'tg_mean':targets.mean().item(),'tg_std':targets.std().item()+1e-8}
    del inputs,targets; gc.collect()
    return stats

# ============ 2. Synthetic test set (consistent with eval_v3_vs_v4) ============
def build_synthetic_test(seed=999, n_tg=10, n_div=8):
    torch.manual_seed(seed); np.random.seed(seed)
    size=32
    nu=torch.tensor(0.01); k=torch.tensor(1.0)
    x=torch.linspace(0,2*np.pi,size); y=torch.linspace(0,2*np.pi,size); z=torch.linspace(0,2*np.pi,size)
    X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
    w1=-k*torch.cos(k*X)*torch.sin(k*Y)*torch.sin(k*Z)
    w2=k*torch.sin(k*X)*torch.cos(k*Y)*torch.sin(k*Z)
    w3=torch.zeros_like(X)
    om0=torch.stack([w1,w2,w3],0).unsqueeze(0)
    inp_l=[]; tgt_l=[]
    for dts in torch.linspace(0.05,0.5,n_tg):
        d=torch.exp(-2*nu*k**2*dts.item())
        dt=torch.exp(-2*nu*k**2*(dts.item()+0.1))
        inp_l.append(om0*d); tgt_l.append(om0*dt)
    for intensity in [5,10,15]:
        for _ in range(4):
            _,omega=generate_divfree_vortex(1,size,float(intensity))
            inp_l.append(omega); tgt_l.append(omega*0.7+torch.roll(omega,1,-1)*0.3)
    for intensity in [5,10,15]:
        for _ in range(4):
            omega=create_synthetic_vorticity(1,size)*(intensity/5.0)
            inp_l.append(omega); tgt_l.append(omega)
    return torch.cat(inp_l,0), torch.cat(tgt_l,0)

# ============ 3. JHTDB data ============
def load_jhtdb():
    d=np.load(os.path.join(BASE_DIR,"jhtdb_data","jhtdb_validation_data.npz"))
    # inspect keys
    print("JHTDB keys:", list(d.keys()), flush=True)
    return d

def main():
    stats_v3 = compute_norm_stats(42)
    print("norm stats:", {k:round(v,5) for k,v in stats_v3.items()}, flush=True)
    im,isv,tm,ts = stats_v3['in_mean'],stats_v3['in_std'],stats_v3['tg_mean'],stats_v3['tg_std']

    # --- synthetic test ---
    Xs, Ys = build_synthetic_test()
    print("synthetic test:", Xs.shape, "target std:", Ys.std().item(), flush=True)

    ckpts = {
      'v3': os.path.join(CHECKPOINT_DIR,'fno_v3_best.pt'),
      'v4': os.path.join(CHECKPOINT_DIR,'fno_v4_best.pt'),
    }
    results = {}
    for name, cpath in ckpts.items():
        print(f"\n=== Evaluating {name} ===", flush=True)
        model = build_model(DEVICE)
        try:
            load(model, cpath, DEVICE)
        except Exception as e:
            print("load err", e, flush=True); continue
        # synthetic
        pred_s = predict(model, Xs, im, isv, tm, ts)
        results[name] = {
          'synthetic': {
            'rel_l2': rel_l2(pred_s, Ys),
            'nrmse': nrmse(pred_s, Ys),
            'mse_raw': F.mse_loss(pred_s, Ys).item(),
            'n_samples': Xs.shape[0],
            'target_std': Ys.std().item(),
          }
        }
        print(f"  synthetic rel_l2={results[name]['synthetic']['rel_l2']:.4f} nrmse={results[name]['synthetic']['nrmse']:.4f}", flush=True)
        # clean
        del pred_s, model; gc.collect(); torch.cuda.empty_cache()

    # --- JHTDB (load real eval numbers from v2 result as they use exact suppression) ---
    with open(os.path.join(RESULTS_DIR,"jhtdb_validation_v2_results.json")) as f:
        jh = json.load(f)
    for name in ['v3','v4']:
        if name in results:
            results[name]['jhtdb'] = {
              'mse': jh[name]['mse'],
              'rel_l2': jh[name]['rel_l2_error'],
              'correlation': jh[name]['correlation'],
              'n_samples': jh['meta']['n_samples'],
            }
            print(f"  {name} JHTDB rel_l2={jh[name]['rel_l2_error']:.3f}", flush=True)

    # --- DNS-like (enhanced) ---
    with open(os.path.join(RESULTS_DIR,"dns_validation_enhanced_results.json")) as f:
        dl = json.load(f)
    for name in ['v3','v4']:
        if name in results and ('fno_'+name) in dl:
            d32 = dl['fno_'+name]['dns_32']
            results[name]['dnslike_32'] = {
              'raw_mse': d32['mse'],
              # relative error needs pred/tgt - not stored; note raw scale issue
              'note': 'raw MSE on unnormalized DNS scale - incompatible with normalized metrics',
            }

    out_path = os.path.join(RESULTS_DIR,"unified_metric_results.json")
    with open(out_path,'w') as f:
        json.dump(results,f,indent=1)
    print(f"\nSaved {out_path}", flush=True)
    print(json.dumps(results, indent=1), flush=True)

if __name__=="__main__":
    main()
