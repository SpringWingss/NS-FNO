#!/usr/bin/env python3
"""
方案B — 真实 NS 训练语料批量生成 (pseudo-spectral DNS)
=====================================================
Generates genuine Navier-Stokes training pairs (omega(t) -> omega(t+dt)) by
running the GPU pseudo-spectral DNS from diverse random divergence-free ICs.

This directly addresses reviewer concerns #3/#5/#6: the current training labels
are analytic/heuristic; here we produce REAL NS-solver trajectories.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 gen_dns_corpus.py --n_traj 8 --snap stride \
      --out $BASE/data/dns_corpus.npz

Each trajectory: random low-k div-free IC, evolved to T, snapshots every dt_save.
Pairing: input = omega(t_n), target = omega(t_n + delta) (fixed NS dt).
Saves velocity/vorticity/energy + metadata to npz.
"""
import os, sys, time, json, argparse, gc
import numpy as np
import torch
sys.path.insert(0, '/mnt/pfs/navier-stokes')
import dns_solver_torch as D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_traj', type=int, default=8)
    ap.add_argument('--N', type=int, default=64, help='grid resolution')
    ap.add_argument('--nu', type=float, default=0.01)
    ap.add_argument('--T', type=float, default=6.0, help='total sim time')
    ap.add_argument('--dt', type=float, default=0.002, help='DNS timestep')
    ap.add_argument('--dt_save', type=float, default=0.1, help='snapshot interval')
    ap.add_argument('--dt_pair', type=float, default=0.1, help='NS dt between input-target (must be multiple of dt_save)')
    ap.add_argument('--k0', type=float, default=2.5, help='peak wavenumber of IC spectrum')
    ap.add_argument('--sigma', type=float, default=1.5, help='width of IC spectral envelope')
    ap.add_argument('--E_target', type=float, default=1.0)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--out', type=str, default='/mnt/pfs/navier-stokes/data/dns_corpus_v1.npz')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[DNS corpus] N={args.N} nu={args.nu} T={args.T} dt={args.dt}", flush=True)
    print(f"  snapshots every dt={args.dt_save}, pairs with NS-dt={args.dt_pair}", flush=True)
    print(f"  device={dev}", flush=True)

    n_snap = int(args.T/args.dt_save) + 1
    step_pair = int(round(args.dt_pair/args.dt_save))

    inp_all, tgt_all = [], []
    meta = []
    en_all = []
    t_start = time.time()
    for tr in range(args.n_traj):
        seed = args.seed0 + tr
        uhat0 = D.random_divfree_ic(args.N, dev, E_target=args.E_target,
                                    k0=args.k0, sigma=args.sigma, seed=seed)
        snaps, times = D.gen_trajectory(uhat0, N=args.N, nu=args.nu, T=args.T,
                                        dt=args.dt, n_snap=n_snap, device=str(dev),
                                        seed=seed)
        # enstrophy trend
        en = [0.5*float((s.astype(np.float64)**2).sum())/(args.N**6) for s in snaps]
        en_all.append(en)
        # build pairs: consecutive snapshots separated by step_pair
        npairs = 0
        for i in range(n_snap - step_pair):
            inp_all.append(snaps[i]); tgt_all.append(snaps[i+step_pair]); npairs+=1
        meta.append({'traj':tr,'seed':seed,'E0':args.E_target,'n_snap':n_snap,'npairs':npairs})
        el=time.time()-t_start
        eta=(el/(tr+1))*(args.n_traj-tr-1)
        print(f"  [{tr+1}/{args.n_traj}] seed={seed} npairs={npairs} |om|max={snaps.max():.2f} "
              f"enstrophy[0..-1]=[{en[0]:.4f}..{en[-1]:.4f}] eta={eta:.0f}s", flush=True)

    X = np.stack(inp_all); Y = np.stack(tgt_all)
    print(f"\n  total pairs: {X.shape} (traj x time...) -> flattened {X.shape[0]}", flush=True)
    print(f"  X range [{X.min():.4f},{X.max():.4f}], Y range [{Y.min():.4f},{Y.max():.4f}]", flush=True)
    np.savez_compressed(args.out, omega_input=X, omega_target=Y,
                        metadata=np.array(meta), energy=np.array(en_all),
                        N=args.N, nu=args.nu, T=args.T, dt=args.dt,
                        dt_save=args.dt_save, dt_pair=args.dt_pair)
    sz=os.path.getsize(args.out)/1024**2
    print(f"  saved {args.out} ({sz:.1f} MB) in {time.time()-t_start:.0f}s", flush=True)

if __name__=='__main__':
    main()
