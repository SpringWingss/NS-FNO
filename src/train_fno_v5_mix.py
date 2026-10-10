#!/usr/bin/env python3
"""
方案B — 混合分辨率 (32³+64³) 多seed 训练
=========================================
同一 VorticityFNO3D 架构同时用 32³ 和 64³ 真实 DNS 语料训练,
每 batch 交替采样两种分辨率。FNO 分辨率无关 → 模型同时学习
低/高分辨率物理规律, 验证"多分辨率联合训练"是否比单分辨率更好。

数据:
  32³: /mnt/mooncake-data/nsfno/data32/v5_seg*.npz (按轨迹 split)
  64³: /mnt/mooncake-data/nsfno/data/v564_*.npy (已 z-score, 含 train/val/test)
归一化: 两者 in_std=0.3068 一致 (同一生成体系), 直接用归一化数据。

用法 (每 GPU 一个 seed):
  CUDA_VISIBLE_DEVICES=$i python3 train_fno_v5_mix.py --seed $S --gpu_id $i --mix_frac 0.5
"""
import os, sys, time, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/mnt/mooncake-data/nsfno')
from train_fno_v3 import VorticityFNO3D
from vorticity_analyzer import EnhancedPhysicsLoss

PHYS_W = {'div': 1.0, 'bkm': 0.5, 'smooth': 0.1, 'enstrophy': 0.1,
          'stretch': 0.1, 'ps': 0.3}
SEEDS = [42, 123, 456, 789, 1111, 2222, 3333, 4444]
DATA32 = '/mnt/mooncake-data/nsfno/data32'
DATA64 = '/mnt/mooncake-data/nsfno/data'


def compute_velocity_from_vorticity(omega):
    B, C, D, H, W = omega.shape
    device = omega.device
    kx = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1, 1) * 2 * np.pi
    ky = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W, 1) * 2 * np.pi
    kz = torch.fft.fftfreq(D, device=device).view(1, 1, 1, 1, D) * 2 * np.pi
    omega_perm = omega.permute(0, 1, 3, 4, 2)
    omega_hat = torch.fft.rfftn(omega_perm, dim=[2, 3, 4])
    Hf, Wf, Df = omega_hat.shape[2], omega_hat.shape[3], omega_hat.shape[4]
    kx_c = kx[:, :, :Hf, :, :Df]; ky_c = ky[:, :, :, :Wf, :Df]; kz_c = kz[:, :, :, :, :Df]
    k_sq = kx_c**2 + ky_c**2 + kz_c**2 + 1e-10
    psi_hat = omega_hat / k_sq
    psi_x = psi_hat[:, 0:1]; psi_y = psi_hat[:, 1:2]; psi_z = psi_hat[:, 2:3]
    ux_hat = -(1j * ky_c * psi_z - 1j * kz_c * psi_y)
    uy_hat = -(1j * kz_c * psi_x - 1j * kx_c * psi_z)
    uz_hat = -(1j * kx_c * psi_y - 1j * ky_c * psi_x)
    ux = torch.fft.irfftn(ux_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uy = torch.fft.irfftn(uy_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uz = torch.fft.irfftn(uz_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    return torch.cat([ux, uy, uz], dim=1)


def load_corpus32(segdir, prefix='v5_seg'):
    """32³ seg npz -> X, Y (already physical, later z-score), traj_ids global."""
    files = sorted(glob.glob(os.path.join(segdir, f'{prefix}*.npz')))
    Xs, Ys, trajs = [], [], []
    base = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X = d['omega_input']; Y = d['omega_target']
        md = d['metadata']
        n = 0
        for m in md:
            np_ = m['npairs'] if isinstance(m, dict) else m.item()['npairs']
            trajs.extend([base + n] * np_)
            n += 1
        base += len(md)
        Xs.append(X); Ys.append(Y)
        print(f"  {os.path.basename(fp)}: {X.shape[0]} pairs, {len(md)} trajs", flush=True)
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0), np.array(trajs, np.int64)


def split_by_traj(traj_ids, train_frac=0.70, val_frac=0.15, seed=42):
    """Split by trajectory to prevent leakage."""
    n = len(traj_ids)
    unique = np.unique(traj_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_tr = int(len(perm) * train_frac); n_va = int(len(perm) * val_frac)
    tr_t = set(perm[:n_tr]); va_t = set(perm[n_tr:n_tr + n_va]); te_t = set(perm[n_tr + n_va:])
    tr_idx = np.array([i for i in range(n) if traj_ids[i] in tr_t])
    va_idx = np.array([i for i in range(n) if traj_ids[i] in va_t])
    te_idx = np.array([i for i in range(n) if traj_ids[i] in te_t])
    return tr_idx, va_idx, te_idx


def main():
    os.environ['PYTHONUNBUFFERED'] = '1'
    torch.set_num_threads(16); os.environ['OMP_NUM_THREADS'] = '16'
    torch.set_num_interop_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=12, help='per-resolution batch')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--modes_h', type=int, default=12)
    ap.add_argument('--modes_w', type=int, default=12)
    ap.add_argument('--modes_d', type=int, default=8)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--decay', type=float, default=1e-4)
    args = ap.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    OUT_BASE = f'{DATA64}/../checkpoints/mix_seed{args.seed}'
    LOG = f'/tmp/train_mix_seed{args.seed}.log'
    sys.stdout = open(LOG, 'w')
    print(f"[mix seed {args.seed}][GPU{args.gpu_id}] device {device}",
          flush=True)

    # ---- 32³ 数据: 加载 + 轨迹split + z-score ----
    print("== loading 32³ corpus ==", flush=True)
    X32, Y32, tid32 = load_corpus32(DATA32)
    tr32, va32, te32 = split_by_traj(tid32, seed=args.seed)
    # z-score using same global stats as v564 (in_std=0.3068...)
    stats32 = json.load(open(f'{DATA64}/v564_stats.json'))
    im, isv, tm, ts = stats32['in_mean'], stats32['in_std'], stats32['tg_mean'], stats32['tg_std']
    X32 = ((X32 - im) / isv).astype(np.float32)
    Y32 = ((Y32 - tm) / ts).astype(np.float32)
    print(f"  32³ train={len(tr32)} val={len(va32)} test={len(te32)}", flush=True)

    # ---- 64³ 数据: 加载 (已z-score) ----
    print("== loading 64³ corpus ==", flush=True)
    X64tr = np.load(f'{DATA64}/v564_X_train.npy'); Y64tr = np.load(f'{DATA64}/v564_Y_train.npy')
    X64va = np.load(f'{DATA64}/v564_X_val.npy');  Y64va = np.load(f'{DATA64}/v564_Y_val.npy')
    X64te = np.load(f'{DATA64}/v564_X_test.npy'); Y64te = np.load(f'{DATA64}/v564_Y_test.npy')
    print(f"  64³ train={X64tr.shape[0]} val={X64va.shape[0]} test={X64te.shape[0]}", flush=True)

    # ---- 模型 + 物理损失 ----
    model = VorticityFNO3D(in_c=3, out_c=3, width=args.width, modes_h=args.modes_h,
                           modes_w=args.modes_w, modes_d=args.modes_d,
                           n_layers=args.n_layers).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    phys = EnhancedPhysicsLoss(viscosity=0.01, lambda_div=PHYS_W['div'],
                               lambda_bkm=PHYS_W['bkm'], lambda_smooth=PHYS_W['smooth'],
                               lambda_enstrophy=PHYS_W['enstrophy'], lambda_ps=PHYS_W['ps'])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def physics_term(pred):
        vel = compute_velocity_from_vorticity(pred)
        _, ld = phys(vel, pred)
        def g(k):
            v = ld[k]
            return v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        l = (g('divergence') * PHYS_W['div'] + g('bkm') * PHYS_W['bkm'] +
             g('smoothness') * PHYS_W['smooth'] + g('enstrophy') * PHYS_W['enstrophy'] +
             g('prodi_serrin') * PHYS_W['ps'])
        if 'stretching' in ld:
            l = l + g('stretching') * PHYS_W['stretch']
        return l

    # ---- 混合训练循环 ----
    g32 = torch.Generator().manual_seed(args.seed)
    g64 = torch.Generator().manual_seed(args.seed + 1)
    best_val = float('inf'); best_epoch = -1
    t0 = time.time(); history = []
    bs = args.batch_size
    # 每 epoch 迭代次数 = max(两分辨率 batch 数), 交替采样
    n32_epoch = len(tr32) // bs
    n64_epoch = len(X64tr) // bs
    n_iter = max(n32_epoch, n64_epoch)
    print(f"epochs={args.epochs} iter/epoch={n_iter} (32³:{n32_epoch} batches, "
          f"64³:{n64_epoch} batches)", flush=True)

    # pre-shuffle indices per epoch (deterministic per seed)
    rng32 = np.random.default_rng(args.seed)
    rng64 = np.random.default_rng(args.seed + 1)
    for epoch in range(args.epochs):
        model.train()
        tl = 0.0; nb = 0
        perm32 = rng32.permutation(tr32); perm64 = rng64.permutation(X64tr.shape[0])
        for it in range(n_iter):
            use32 = (it % 2 == 0)
            if use32:
                i0 = (it // 2) % n32_epoch
                idx = perm32[i0 * bs:(i0 + 1) * bs]
                x = torch.from_numpy(X32[idx]).to(device)
                y = torch.from_numpy(Y32[idx]).to(device)
            else:
                i0 = (it // 2) % n64_epoch
                idx = perm64[i0 * bs:(i0 + 1) * bs]
                x = torch.from_numpy(X64tr[idx]).to(device)
                y = torch.from_numpy(Y64tr[idx]).to(device)
            if x.shape[0] < 2:
                continue
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            with torch.no_grad():
                ph = physics_term(pred[:4])
            loss = loss + ph if torch.isfinite(ph) else loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        avg_tl = tl / max(nb, 1)

        model.eval()
        # val on 64³ (primary metric, same as pure-64 baseline)
        v64 = 0.0; vn = 0
        with torch.no_grad():
            for i in range(0, len(X64va), 2 * bs):
                x = torch.from_numpy(X64va[i:i + 2 * bs]).to(device)
                y = torch.from_numpy(Y64va[i:i + 2 * bs]).to(device)
                v64 += F.mse_loss(model(x), y).item(); vn += 1
        # val on 32³
        v32 = 0.0
        with torch.no_grad():
            for i in range(0, len(va32), 2 * bs):
                x = torch.from_numpy(X32[va32[i:i + 2 * bs]]).to(device)
                y = torch.from_numpy(Y32[va32[i:i + 2 * bs]]).to(device)
                v32 += F.mse_loss(model(x), y).item()
        avg_v = (v64 + v32) / (vn + max(1, len(va32) // (2 * bs)))
        sched.step()

        history.append({'epoch': epoch, 'train_loss': avg_tl,
                        'val_64': v64 / max(vn, 1), 'val_32': v32 / max(1, len(va32) // (2 * bs)),
                        'val_avg': avg_v})
        if avg_v < best_val:
            best_val = avg_v; best_epoch = epoch
            torch.save({'epoch': epoch, 'seed': args.seed,
                        'model_state_dict': model.state_dict(), 'val_loss': best_val,
                        'norm_stats': {k: stats32[k] for k in ('in_mean', 'in_std', 'tg_mean', 'tg_std')},
                        'config': vars(args), 'resolution': 'mix32_64',
                        'val_64': v64 / max(vn, 1), 'val_32': v32 / max(1, len(va32) // (2 * bs))},
                       f'{OUT_BASE}_best.pt')
        el = time.time() - t0
        print(f"[mix {args.seed}] Ep {epoch+1:3d}/{args.epochs}: train={avg_tl:.6f} "
              f"val64={v64/max(vn,1):.6f} val32={v32/max(1,len(va32)//(2*bs)):.6f} "
              f"lr={sched.get_last_lr()[0]:.2e} ({el:.0f}s)", flush=True)

    torch.save({'model_state_dict': model.state_dict(), 'history': history,
                'best_epoch': best_epoch, 'seed': args.seed,
                'norm_stats': {k: stats32[k] for k in ('in_mean', 'in_std', 'tg_mean', 'tg_std')},
                'config': vars(args), 'resolution': 'mix32_64'},
               f'{OUT_BASE}_final.pt')
    print(f"[mix {args.seed}] DONE best val={best_val:.6f} (ep {best_epoch})", flush=True)


if __name__ == '__main__':
    main()
