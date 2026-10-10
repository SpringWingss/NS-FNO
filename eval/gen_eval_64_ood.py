#!/usr/bin/env python3
"""
方案A — 64³ DNS-OOD 测试集生成 + 64³ 网格评估
==============================================
在 64³ 网格上用真实 DNS 伪谱求解器生成 OOD 测试对:

  T1 time:     E_target=E_id, dt_pair∈{0.1(ID),0.2,0.5,1.0}
  T2 intensity:E_target∈{0.04(ID),0.2,0.35}, dt_pair=0.1
  T3 resolution: 本身就是 64³ 全分辨率 (无下采样), 用于对比 32³ OOD

产出: eval_data/v5_ood_64_time.pt / v5_ood_64_intensity.pt / v5_ood_64_resolution.pt
评估: 用 64³ V5 8-seed 模型 + 32³ V5 基线, 输出各维度 rel-L2 对比 → 印证"64³ 模型
      在其原生分辨率上的真实能力, 以及 32³ 低分辨率训练在 64³ 网格上的失配"

用法 (每GPU一个分块或单卡):
  python3 gen_eval_64_ood.py --out_dir eval_data
"""
import os, sys, time, json, gc, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/mnt/mooncake-data/nsfno')
import dns_ns3d as D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def curl_np(s):
    """curl of velocity snap (3,D,H,W numpy) -> vorticity."""
    wx = np.gradient(s[2], axis=1) - np.gradient(s[1], axis=2)
    wy = np.gradient(s[0], axis=2) - np.gradient(s[2], axis=0)
    wz = np.gradient(s[1], axis=0) - np.gradient(s[0], axis=1)
    return np.stack([wx, wy, wz], 0)


def run_traj(N, nu, dt, dt_save, T, E_target, k0, sigma, seed, dev, max_retry=5):
    """Run one DNS trajectory, return list of physical velocity snaps (float32 cpu)."""
    solver = D.NS3D(N=N, nu=nu, device=str(dev))
    KX, KY, KZ, K2, _ = D.make_grid(N, dev)
    n_snap = int(T / dt_save) + 1
    for att in range(max_retry):
        uphys = D.random_divfree_ic(N, dev, E_target=E_target, k0=k0, sigma=sigma,
                                    seed=seed + att * 997)
        uh = torch.fft.rfftn(uphys, dim=(1, 2, 3))
        uh = D.project_divfree(uh, KX, KY, KZ, K2)
        uh = D.dealias(uh, KX, KY, KZ, N)
        snaps = []
        bad = False
        for i in range(n_snap):
            s = solver.to_physical(uh).cpu().float().numpy()
            if not np.isfinite(s).all():
                bad = True; break
            snaps.append(s)
            if i < n_snap - 1:
                for _ in range(int(dt_save / dt)):
                    uh = solver.step(uh, dt)
                    uh = D.project_divfree(uh, KX, KY, KZ, K2)
                    uh = D.dealias(uh, KX, KY, KZ, N)
                    if not torch.isfinite(uh).all():
                        bad = True; break
                if bad:
                    break
        if not bad:
            return snaps
        gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError(f"traj seed {seed} failed {max_retry} times")


def gen_time_64(n_traj=8, N=64, nu=0.01, dt=2e-4, dt_save=0.05,
                dt_pairs=(0.1, 0.2, 0.5, 1.0), E_target=0.08, seed0=9100):
    """64³ time-extrapolation OOD pairs."""
    dev = DEVICE
    T = 2.0
    sp = {dp: max(1, int(round(dp / dt_save))) for dp in dt_pairs}
    out = {dp: {'X': [], 'Y': []} for dp in dt_pairs}
    for tr in range(n_traj):
        snaps = run_traj(N, nu, dt, dt_save, T, E_target, 2.5, 1.5, seed0 + tr, dev)
        n = len(snaps)
        print(f"  [time] traj {tr}: {n} snaps", flush=True)
        for dp in dt_pairs:
            k = sp[dp]
            for i in range(0, n - k, 4):
                w0 = curl_np(snaps[i]); w1 = curl_np(snaps[i + k])
                out[dp]['X'].append(w0); out[dp]['Y'].append(w1)
    res = {}
    for dp, v in out.items():
        res[dp] = (torch.from_numpy(np.stack(v['X'])).float(),
                   torch.from_numpy(np.stack(v['Y'])).float())
        print(f"  time dt={dp}: {v['X'][0].shape} x{len(v['X'])}", flush=True)
    return res


def gen_intensity_64(intensities=(0.04, 0.2, 0.35), n_traj=6, N=64, nu=0.01,
                     dt=2e-4, dt_save=0.05, dt_pair=0.1, seed0=8200):
    """64³ intensity-extrapolation OOD pairs (higher E_target)."""
    dev = DEVICE
    sp = max(1, int(round(dt_pair / dt_save)))
    out = {}
    for et in intensities:
        Xs, Ys = [], []
        T = 2.0 if et <= 0.1 else 1.5
        for tr in range(n_traj):
            seed = seed0 + int(et * 100) + tr
            try:
                snaps = run_traj(N, nu, dt, dt_save, T, et, 2.5, 1.5, seed, dev)
            except RuntimeError as e:
                print(f"  [intensity] E={et} traj {tr}: {e}", flush=True); continue
            n = len(snaps)
            for i in range(0, n - sp, 3):
                w0 = curl_np(snaps[i]); w1 = curl_np(snaps[i + sp])
                Xs.append(w0); Ys.append(w1)
        out[et] = (torch.from_numpy(np.stack(Xs)).float() if Xs else torch.zeros(0, 3, N, N, N),
                   torch.from_numpy(np.stack(Ys)).float() if Ys else torch.zeros(0, 3, N, N, N))
        print(f"  intensity E={et}: {len(Xs)} pairs", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='/mnt/mooncake-data/nsfno/eval_data')
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--n_traj_time', type=int, default=8)
    ap.add_argument('--n_traj_inten', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {args.device} | N={args.N}", flush=True)
    t0 = time.time()

    print("\n== 64³ time OOD ==", flush=True)
    t_ood = gen_time_64(n_traj=args.n_traj_time, N=args.N)
    torch.save(t_ood, os.path.join(args.out_dir, 'v5_ood_64_time.pt'))
    print(f"  saved v5_ood_64_time.pt ({time.time()-t0:.0f}s)", flush=True)

    print("\n== 64³ intensity OOD ==", flush=True)
    i_ood = gen_intensity_64(n_traj=args.n_traj_inten, N=args.N)
    torch.save(i_ood, os.path.join(args.out_dir, 'v5_ood_64_intensity.pt'))
    print(f"  saved v5_ood_64_intensity.pt ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
