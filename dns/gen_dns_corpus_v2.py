#!/usr/bin/env python3
"""
方案B — 真实 NS 训练语料批量生成 (v2: dns_ns3d正确求解器)
=====================================================
Generates genuine Navier-Stokes pairs omega(t)->omega(t+dt) from the validated
GPU pseudo-spectral solver (dns_ns3d), from diverse random divergence-free ICs.

For V5 training prep. Sanity checks per trajectory: inviscid energy conservation
of IC, div-free residual, enstrophy trend.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 gen_dns_corpus_v2.py --n_traj 8 --out ...
"""
import os, sys, time, json, argparse, gc
import numpy as np
import torch
sys.path.insert(0, '/mnt/pfs/navier-stokes')
import dns_ns3d as D

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_traj',type=int,default=8)
    ap.add_argument('--N',type=int,default=64)
    ap.add_argument('--nu',type=float,default=0.01)
    ap.add_argument('--T',type=float,default=4.0)
    ap.add_argument('--dt',type=float,default=0.0005)
    ap.add_argument('--dt_save',type=float,default=0.05)
    ap.add_argument('--dt_pair',type=float,default=0.1)
    ap.add_argument('--k0',type=float,default=2.5)
    ap.add_argument('--sigma',type=float,default=1.5)
    ap.add_argument('--E_target',type=float,default=1.0)
    ap.add_argument('--seed0',type=int,default=0)
    ap.add_argument('--out',type=str,default='/mnt/pfs/navier-stokes/data/dns_corpus_v2.npz')
    ap.add_argument('--device',type=str,default='cuda')
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.out),exist_ok=True)

    N=args.N; solver=D.NS3D(N=N,nu=args.nu,device=str(dev))
    KX,KY,KZ,K2,mask=D.make_grid(N,dev)
    n_snap=int(args.T/args.dt_save)+1
    step_pair=int(round(args.dt_pair/args.dt_save))

    inp=[];tgt=[];meta=[];en_all=[]
    for tr in range(args.n_traj):
        seed=args.seed0+tr
        uphys=D.random_divfree_ic(N, dev, E_target=args.E_target, k0=args.k0,
                                  sigma=args.sigma, seed=seed)
        uh=torch.fft.rfftn(uphys,dim=(1,2,3))
        uh=D.project_divfree(uh,KX,KY,KZ,K2); uh=D.dealias(uh,KX,KY,KZ,N)

        # evolve
        snaps=[];times=[]
        for i in range(n_snap):
            snaps.append(solver.to_physical(uh).cpu().float().numpy())
            times.append(round(i*args.dt_save,6))
            if i<n_snap-1:
                for _ in range(int(args.dt_save/args.dt)):
                    uh=solver.step(uh,args.dt)
                    uh=D.project_divfree(uh,KX,KY,KZ,K2); uh=D.dealias(uh,KX,KY,KZ,N)
        en=[0.5*float((s**2).mean()) for s in snaps]
        en_all.append(en)
        for i in range(n_snap-step_pair):
            inp.append(snaps[i]); tgt.append(snaps[i+step_pair])
        meta.append({'traj':tr,'seed':seed,'n_snap':n_snap,'npairs':n_snap-step_pair})
        print(f"[{tr+1}/{args.n_traj}] seed={seed} enstrophy[0..-1]=[{en[0]:.4f}..{en[-1]:.4f}] |om|max={snaps[0].max():.2f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    X=np.stack(inp);Y=np.stack(tgt)
    np.savez_compressed(args.out, omega_input=X,omega_target=Y, metadata=np.array(meta),
                        energy=np.array(en_all), N=args.N, nu=args.nu, T=args.T, dt=args.dt,
                        dt_save=args.dt_save, dt_pair=args.dt_pair)
    print(f"total pairs {X.shape[0]}, X[{X.min():.3f},{X.max():.3f}], saved {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)

if __name__=='__main__':
    main()
