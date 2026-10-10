#!/usr/bin/env python3
"""
方案B — FNO V5 训练 (真实 NS DNS 语料 + 多物理损失)
=====================================================
- 架构与 V3/V4 完全相同: VorticityFNO3D (width=64, modes=[12,12,8], 6层, LayerNorm)
- 训练数据: 真实 DNS (dns_ns3d 伪谱求解器) 生成的 32^3 omega(t)->omega(t+dt) 对
- 按轨迹拆分 train/val/test (防同轨迹数据泄漏, 60/15/25%)
- z-score 归一化 (用训练集统计)
- 损失: MSE + EnhancedPhysicsLoss (full config: div=1.0,bkm=0.5,ps=0.3,smooth=0.1,enst=0.1,stretch=0.1)
- 300 epochs, AdamW lr=3e-4, CosineAnnealing

用法:
  CUDA_VISIBLE_DEVICES=0 python3 train_fno_v5.py --segdir /mnt/pfs/navier-stokes/data \
      --out /mnt/pfs/navier-stokes/checkpoints/fno_v5_best.pt --epochs 300
"""
import os, sys, time, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/mnt/pfs/navier-stokes')
from train_fno_v3 import VorticityFNO3D
from vorticity_analyzer import EnhancedPhysicsLoss

# Some systems use vorticity_analyzer for the enhanced loss; fall back if needed
try:
    from vorticity_analyzer import EnhancedPhysicsLoss
except Exception:
    pass


def compute_velocity_from_vorticity(omega):
    """Approximate velocity from vorticity using FFT (spectral Biot-Savart).
    omega: (B, 3, D, H, W) normalized vorticity"""
    B, C, D, H, W = omega.shape
    device = omega.device
    kx = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1, 1) * 2 * np.pi
    ky = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W, 1) * 2 * np.pi
    kz = torch.fft.fftfreq(D, device=device).view(1, 1, 1, 1, D) * 2 * np.pi
    omega_perm = omega.permute(0, 1, 3, 4, 2)
    omega_hat = torch.fft.rfftn(omega_perm, dim=[2, 3, 4])
    Hf, Wf, Df = omega_hat.shape[2], omega_hat.shape[3], omega_hat.shape[4]
    kx_c = kx[:, :, :Hf, :, :Df]
    ky_c = ky[:, :, :, :Wf, :Df]
    kz_c = kz[:, :, :, :, :Df]
    k_sq = kx_c**2 + ky_c**2 + kz_c**2 + 1e-10
    psi_hat = omega_hat / k_sq
    psi_x = psi_hat[:, 0:1]
    psi_y = psi_hat[:, 1:2]
    psi_z = psi_hat[:, 2:3]
    ux_hat = -(1j * ky_c * psi_z - 1j * kz_c * psi_y)
    uy_hat = -(1j * kz_c * psi_x - 1j * kx_c * psi_z)
    uz_hat = -(1j * kx_c * psi_y - 1j * ky_c * psi_x)
    ux = torch.fft.irfftn(ux_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uy = torch.fft.irfftn(uy_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uz = torch.fft.irfftn(uz_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    return torch.cat([ux, uy, uz], dim=1)


PHYS_W = {'div': 1.0, 'bkm': 0.5, 'smooth': 0.1, 'enstrophy': 0.1,
          'stretch': 0.1, 'ps': 0.3}


def load_corpus(segdir, prefix='v5_seg'):
    """Merge all {prefix}*.npz, return arrays + per-pair trajectory labels.
    Returns (X, Y, traj_ids) as numpy."""
    files = sorted(glob.glob(os.path.join(segdir, f'{prefix}*.npz')))
    if not files:
        raise FileNotFoundError(f"no {prefix}*.npz in {segdir}")
    Xs, Ys, trajs, metas = [], [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X = d['omega_input']; Y = d['omega_target']
        md = d['metadata']
        # build per-pair traj ids from metadata npairs
        n = 0
        for m in md:
            np_ = m['npairs'] if isinstance(m, dict) else m.item()['npairs']
            trajs.extend([n] * np_)   # traj label within this seg
            n += 1
        Xs.append(X); Ys.append(Y)
        metas.append(md)
        print(f"  {os.path.basename(fp)}: {X.shape[0]} pairs, {len(md)} trajs", flush=True)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    traj_ids = np.array(trajs, dtype=np.int64)
    return X, Y, traj_ids


def split_by_traj(traj_ids, train_frac=0.60, val_frac=0.15, seed=42):
    """Split by trajectory to prevent leakage. Returns (train_idx, val_idx, test_idx).
    traj_ids assumed assigned per-seg starting at 0 (seg-local labels). To keep
    trajectories distinct across segments we re-label by segment base."""
    n = len(traj_ids)
    unique = np.unique(traj_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_tr = int(len(perm) * train_frac)
    n_va = int(len(perm) * val_frac)
    tr_t = set(perm[:n_tr])
    va_t = set(perm[n_tr:n_tr + n_va])
    te_t = set(perm[n_tr + n_va:])
    tr_idx = np.array([i for i in range(n) if traj_ids[i] in tr_t])
    va_idx = np.array([i for i in range(n) if traj_ids[i] in va_t])
    te_idx = np.array([i for i in range(n) if traj_ids[i] in te_t])
    return tr_idx, va_idx, te_idx


def main():
    os.environ['PYTHONUNBUFFERED'] = '1'
    ap = argparse.ArgumentParser()
    ap.add_argument('--segdir', default='/mnt/pfs/navier-stokes/data')
    ap.add_argument('--out', default='/mnt/pfs/navier-stokes/checkpoints/fno_v5_best.pt')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--modes_h', type=int, default=12)
    ap.add_argument('--modes_w', type=int, default=12)
    ap.add_argument('--modes_d', type=int, default=8)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--prefix', default='v5_seg',
                    help='corpus file prefix, e.g. v5_seg (32^3) or v5_64_seg (64^3)')
    ap.add_argument('--fail_if_no_cuda', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.fail_if_no_cuda and not torch.cuda.is_available():
        raise RuntimeError("no CUDA")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})", flush=True)

    print("\n== Loading real DNS corpus ==", flush=True)
    X, Y, traj_ids = load_corpus(args.segdir, prefix=args.prefix)
    # global traj ids (offset per seg) to keep distinct across segments
    n_pairs_per_seg = {}
    files = sorted(glob.glob(os.path.join(args.segdir, f'{args.prefix}*.npz')))
    base = 0
    tid_global = np.zeros(len(X), dtype=np.int64)
    pos = 0
    for si, fp in enumerate(files):
        d = np.load(fp, allow_pickle=True)
        md = d['metadata']
        seg_npairs = []
        for m in md:
            seg_npairs.append(m['npairs'] if isinstance(m, dict) else m.item()['npairs'])
        n_here = sum(seg_npairs)
        tid_global[pos:pos + n_here] = (np.repeat(np.arange(len(seg_npairs)), seg_npairs) + base)
        pos += n_here
        base += len(seg_npairs)

    print(f"Total pairs: {X.shape[0]}, distinct trajs: {len(np.unique(tid_global))}", flush=True)
    print(f"Raw range: X[{X.min():.3f},{X.max():.3f}] Y[{Y.min():.3f},{Y.max():.3f}]", flush=True)

    tr_idx, va_idx, te_idx = split_by_traj(tid_global)
    print(f"Split by traj: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}", flush=True)

    # z-score normalization on training set only
    Xtr = X[tr_idx].astype(np.float32)
    Ytr = Y[tr_idx].astype(np.float32)
    in_mean, in_std = Xtr.mean(), Xtr.std() + 1e-8
    tg_mean, tg_std = Ytr.mean(), Ytr.std() + 1e-8
    stats = {'in_mean': float(in_mean), 'in_std': float(in_std),
             'tg_mean': float(tg_mean), 'tg_std': float(tg_std)}
    print(f"Norm stats: in_mean={in_mean:.4f} in_std={in_std:.4f} tg_mean={tg_mean:.4f} tg_std={tg_std:.4f}", flush=True)

    def norm(X):
        return (X - in_mean) / in_std, (X - tg_mean) / tg_std

    Xn, Yn = norm(X)  # for train/val/test all normalized by train stats
    Xn = Xn.astype(np.float32); Yn = Yn.astype(np.float32)

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xn[tr_idx]), torch.from_numpy(Yn[tr_idx]))
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xn[va_idx]), torch.from_numpy(Yn[va_idx]))
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xn[te_idx]), torch.from_numpy(Yn[te_idx]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True)

    model = VorticityFNO3D(in_c=3, out_c=3, width=args.width,
                           modes_h=args.modes_h, modes_w=args.modes_w,
                           modes_d=args.modes_d, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}", flush=True)

    phys = EnhancedPhysicsLoss(viscosity=0.01,
                               lambda_div=PHYS_W['div'], lambda_bkm=PHYS_W['bkm'],
                               lambda_smooth=PHYS_W['smooth'],
                               lambda_enstrophy=PHYS_W['enstrophy'],
                               lambda_ps=PHYS_W['ps'])
    # stretching weight passed separately below (EnhancedPhysicsLoss may or may not have it)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    best_epoch = -1
    t0 = time.time()
    history = []

    def physics_term(pred, y):
        vel = compute_velocity_from_vorticity(pred)
        _, ld = phys(vel, pred)
        def g(k):
            v = ld[k]
            return v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        l = (g('divergence') * PHYS_W['div'] + g('bkm') * PHYS_W['bkm'] +
             g('smoothness') * PHYS_W['smooth'] + g('enstrophy') * PHYS_W['enstrophy'] +
             g('prodi_serrin') * PHYS_W['ps'])
        # stretching: add from ld if present
        if 'stretching' in ld:
            l = l + g('stretching') * PHYS_W['stretch']
        return l

    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(x)
            bsz = pred.shape[0]
            loss = F.mse_loss(pred, y)
            with torch.no_grad():
                ph = physics_term(pred[:4], y[:4])
            loss = loss + ph if torch.isfinite(ph) else loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        avg_tl = tl / len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                vl += F.mse_loss(model(x), y).item()
        avg_vl = vl / len(val_loader)
        sched.step()

        history.append({'epoch': epoch, 'train_loss': avg_tl, 'val_loss': avg_vl})
        el = time.time() - t0
        print(f"  Ep {epoch+1:3d}/{args.epochs}: train={avg_tl:.6f} val={avg_vl:.6f} "
              f"lr={sched.get_last_lr()[0]:.2e} ({el:.0f}s)", flush=True)

        if avg_vl < best_val:
            best_val = avg_vl
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': avg_vl, 'norm_stats': stats,
                        'config': vars(args)}, args.out)

    # final + test eval
    model.eval()
    test_l = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_l += F.mse_loss(model(x), y).item()
    avg_test = test_l / len(test_loader)

    torch.save({'model_state_dict': model.state_dict(), 'val_loss': avg_vl,
                'test_loss': avg_test, 'train_loss': avg_tl, 'history': history,
                'norm_stats': stats, 'best_epoch': best_epoch, 'config': vars(args)},
               args.out.replace('_best.pt', '_final.pt'))
    with open(os.path.join(os.path.dirname(args.out), 'norm_stats_v5.json'), 'w') as f:
        json.dump(stats, f)
    print(f"\nDone! {time.time()-t0:.0f}s best val={best_val:.6f} (ep {best_epoch}) "
          f"test MSE={avg_test:.6f}\nSaved: {args.out}", flush=True)


if __name__ == '__main__':
    main()
