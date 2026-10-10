#!/usr/bin/env python3
"""
方案B — #6 物理指标评估: full(multi-physics) vs mse vs nobkm vs nops
====================================================================
Critical reviewer question: does the multi-physics loss improve OOD generalization
and physics consistency over a pure-MSE FNO of equal architecture/budget?

Evaluates ablation checkpoints (from the earlier single-seed run, seed=42) on:
  1. In-distribution synthetic rel-L2
  2. Temporal OOD (direct single-step at larger dt)
  3. Intensity OOD (direct single-step at out-of-range amplitude)
  4. Physics-consistency metrics on predictions:
       - div(u) error (mean |div u|) via Biot-Savart-inverted velocity
       - mean |div omega| (must be ~0 by construction)
       - spectral slope error vs target (effective resolution)
       - per-sample rel-L2 (variance)

Output: results/ablation_physics_metrics.json
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F

BASE="/mnt/pfs/navier-stokes"
sys.path.insert(0, BASE)
from train_fno_v3 import VorticityFNO3D
from vorticity_analyzer import EnhancedPhysicsLoss, create_synthetic_vorticity
from vorticity_fno import generate_divfree_vortex

torch.manual_seed(42); np.random.seed(42)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, flush=True)

def load(ck, device):
    m=VorticityFNO3D(in_c=3,out_c=3,width=64,modes_h=12,modes_w=12,modes_d=8,n_layers=6).to(device)
    c=torch.load(ck, map_location=device)
    sd=c.get('model_state_dict', c); m.load_state_dict(sd); m.eval()
    return m

# reuse training norm stats (recompute deterministically)
def compute_norm_stats(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    inputs,targets=[],[]; size=32
    nu=torch.tensor(0.01); k=torch.tensor(1.0)
    x=torch.linspace(0,2*np.pi,size); y=torch.linspace(0,2*np.pi,size); z=torch.linspace(0,2*np.pi,size)
    X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
    w1=-k*torch.cos(k*X)*torch.sin(k*Y)*torch.sin(k*Z)
    w2=k*torch.sin(k*X)*torch.cos(k*Y)*torch.sin(k*Z)
    w3=torch.zeros_like(X); om0=torch.stack([w1,w2,w3],0).unsqueeze(0)
    for dt1 in torch.linspace(0.05,1.0,10):
        for dt2 in torch.linspace(0.1,2.0,8):
            d=torch.exp(-2*nu*k**2*dt1); dt=torch.exp(-2*nu*k**2*(dt1+dt2))
            inputs.append(om0*d); targets.append(om0*dt)
    for inten in [0.5,1,2,3,5,8,10,12,15,18,20]:
        n=250 if inten<=15 else 50
        for _ in range(n):
            _,om=generate_divfree_vortex(1,size,float(inten))
            ns=0.03*(1.0+inten/20.0); noise=torch.randn_like(om)*ns
            targets.append(om*0.7+torch.roll(om,1,-1)*0.3+noise); inputs.append(om)
    for inten in [0.5,1,2,3,5,8,10,12,15,18,20]:
        n=180 if inten<=15 else 20
        for _ in range(n):
            om=create_synthetic_vorticity(1,size)*(inten/5.0)
            ns=0.03*(1.0+inten/20.0); noise=torch.randn_like(om)*ns
            inputs.append(om*0.65+torch.roll(om,(-1,2),(-2,-1))*0.35+noise); targets.append(om)
    inputs=torch.cat(inputs,0); targets=torch.cat(targets,0)
    return {'in_mean':inputs.mean().item(),'in_std':inputs.std().item()+1e-8,
            'tg_mean':targets.mean().item(),'tg_std':targets.std().item()+1e-8}

def velocity_from_vorticity(omega):
    B,C,D,H,W=omega.shape; dev=omega.device
    kx=torch.fft.fftfreq(H,device=dev).view(1,1,H,1,1)*2*np.pi
    ky=torch.fft.fftfreq(W,device=dev).view(1,1,1,W,1)*2*np.pi
    kz=torch.fft.fftfreq(D,device=dev).view(1,1,1,1,D)*2*np.pi
    perm=omega.permute(0,1,3,4,2); ohat=torch.fft.rfftn(perm,dim=[2,3,4])
    Hf,Wf,Df=ohat.shape[2],ohat.shape[3],ohat.shape[4]
    kxc=kx[:,:,:Hf,:,:Df]; kyc=ky[:,:,:,:Wf,:Df]; kzc=kz[:,:,:,:,:Df]
    ks=kxc**2+kyc**2+kzc**2+1e-10
    psi=ohat/ks; px,py,pz=psi[:,0:1],psi[:,1:2],psi[:,2:3]
    ux=torch.fft.irfftn(-(1j*kyc*pz-1j*kzc*py),s=[H,W,D],dim=[2,3,4]).permute(0,1,4,2,3)
    uy=torch.fft.irfftn(-(1j*kzc*px-1j*kxc*pz),s=[H,W,D],dim=[2,3,4]).permute(0,1,4,2,3)
    uz=torch.fft.irfftn(-(1j*kxc*py-1j*kyc*px),s=[H,W,D],dim=[2,3,4]).permute(0,1,4,2,3)
    return torch.cat([ux,uy,uz],1)

def div_of_u(u):
    u1,u2,u3=u[:,0],u[:,1],u[:,2]
    return (torch.gradient(u1,dim=3)[0]+torch.gradient(u2,dim=2)[0]+torch.gradient(u3,dim=1)[0]).abs().mean().item()

def rel_l2(p,t): return (p-t).norm().item()/(t.norm().item()+1e-8)

@torch.no_grad()
def predict(model, omega, st, bs=32):
    out=[]
    im,isv,tm,ts=st['in_mean'],st['in_std'],st['tg_mean'],st['tg_std']
    for i in range(0,omega.shape[0],bs):
        b=omega[i:i+bs].to(DEVICE)
        p=model((b-im)/isv); out.append((p*ts+tm).cpu())
    return torch.cat(out,0)

def build_synthetic_test(seed=999):
    torch.manual_seed(seed); np.random.seed(seed); size=32
    nu=torch.tensor(0.01); k=torch.tensor(1.0)
    x=torch.linspace(0,2*np.pi,size); y=torch.linspace(0,2*np.pi,size); z=torch.linspace(0,2*np.pi,size)
    X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
    w1=-k*torch.cos(k*X)*torch.sin(k*Y)*torch.sin(k*Z); w2=k*torch.sin(k*X)*torch.cos(k*Y)*torch.sin(k*Z)
    w3=torch.zeros_like(X); om0=torch.stack([w1,w2,w3],0).unsqueeze(0)
    inp,tgt=[],[]
    for dts in torch.linspace(0.05,0.5,10):
        d=torch.exp(-2*nu*k**2*dts.item()); dt=torch.exp(-2*nu*k**2*(dts.item()+0.1))
        inp.append(om0*d); tgt.append(om0*dt)
    for inten in [5,10,15]:
        for _ in range(4):
            _,om=generate_divfree_vortex(1,size,float(inten)); inp.append(om); tgt.append(om*0.7+torch.roll(om,1,-1)*0.3)
    for inten in [5,10,15]:
        for _ in range(4):
            om=create_synthetic_vorticity(1,size)*(inten/5.0); inp.append(om); tgt.append(om)
    return torch.cat(inp,0), torch.cat(tgt,0)

def build_temporal_ood():
    # direct prediction at larger dt (train dt=0.1)
    torch.manual_seed(7); np.random.seed(7); size=32
    nu=torch.tensor(0.01); k=torch.tensor(1.0)
    x=torch.linspace(0,2*np.pi,size); y=torch.linspace(0,2*np.pi,size); z=torch.linspace(0,2*np.pi,size)
    X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
    w1=-k*torch.cos(k*X)*torch.sin(k*Y)*torch.sin(k*Z); w2=k*torch.sin(k*X)*torch.cos(k*Y)*torch.sin(k*Z)
    w3=torch.zeros_like(X); om0=torch.stack([w1,w2,w3],0).unsqueeze(0)
    dts=[0.5,1.0,2.0,5.0]
    sets={}
    for d in dts:
        inp=om0*torch.exp(-2*nu*k**2*0.5)
        tgt=om0*torch.exp(-2*nu*k**2*(0.5+d))
        sets[f'dt{d}']=(inp,tgt)
    return sets

def build_intensity_ood():
    torch.manual_seed(11); np.random.seed(11); size=32
    sets={}
    for inten in [0.05,0.1,25,50]:  # outside training 0.5-20
        om=create_synthetic_vorticity(2,size)*(inten/5.0)
        tgt=om*0.7+torch.roll(om,1,-1)*0.3
        sets[f'A{inten}']=(om,tgt)
    return sets

def spectral_slope_err(pred, tgt):
    def slope(f):
        B,C,D,H,W=f.shape
        fh=torch.fft.rfftn(f.cpu(),dim=[2,3,4])   # (B,C,D,H,Wr) with Wr=W//2+1
        spec=torch.abs(fh).pow(2).sum(1).mean(0)  # (D,H,Wr)
        D,H,Wr=spec.shape
        kz=torch.fft.fftfreq(D); ky=torch.fft.fftfreq(H); kx=torch.fft.rfftfreq(W)
        # spec[d,h,w] corresponds to kz[d], ky[h], kx[w]
        KZ,KY,KX=torch.meshgrid(kz,ky,kx,indexing='ij')
        kmag=torch.sqrt(KZ**2+KY**2+KX**2+1e-12)
        mask=(kmag>1e-3)&(kmag<8)
        kk=kmag[mask].numpy(); ss=spec[mask].numpy()
        kk=kk[kk>0]
        if len(kk)<4: return None
        lk=np.log(kk); ls=np.log(ss+1e-12)
        p=np.polyfit(lk,ls,1)
        return p[0]
    sp=slope(pred); stp=slope(tgt)
    return abs(sp-stp) if (sp is not None and stp is not None) else None

def main():
    st=compute_norm_stats(42)
    ckpts={'full':'checkpoints/ablation_full_best.pt','mse':'checkpoints/ablation_mse_best.pt',
           'nobkm':'checkpoints/ablation_nobkm_best.pt','nops':'checkpoints/ablation_nops_best.pt'}
    # ID test
    Xs,Ys=build_synthetic_test()
    temp_sets=build_temporal_ood(); int_sets=build_intensity_ood()
    results={}
    for name,path in ckpts.items():
        ck=os.path.join(BASE,path)
        if not os.path.exists(ck): print("missing",ck); continue
        model=load(ck,DEVICE)
        print(f"\n=== {name} ===", flush=True)
        # ID
        ps=predict(model,Xs,st); res={'id_rel_l2':rel_l2(ps,Ys)}
        # physics: div u on ID pred
        res['id_divu']=div_of_u(velocity_from_vorticity(ps).to(DEVICE))
        # temporal OOD
        res['temporal']={}
        for k,(i,t) in temp_sets.items():
            p=predict(model,i,st); res['temporal'][k]={'rel_l2':rel_l2(p,t)}
        # intensity OOD
        res['intensity']={}
        for k,(i,t) in int_sets.items():
            p=predict(model,i,st); res['intensity'][k]={'rel_l2':rel_l2(p,t)}
        # spectral
        res['spectral_slope_err']=spectral_slope_err(ps,Ys)
        results[name]=res
        print("  ",json.dumps({k:v for k,v in res.items() if k not in ('temporal','intensity')}),flush=True)
        print("  temporal:",{k:round(v['rel_l2'],4) for k,v in res['temporal'].items()},flush=True)
        print("  intensity:",{k:round(v['rel_l2'],4) for k,v in res['intensity'].items()},flush=True)
        del model,ps; gc.collect(); torch.cuda.empty_cache()
    out=os.path.join(BASE,'results','ablation_physics_metrics.json')
    with open(out,'w') as f: json.dump(results,f,indent=1)
    print("\nSaved",out,flush=True)

if __name__=='__main__':
    main()
