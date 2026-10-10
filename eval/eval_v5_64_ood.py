#!/usr/bin/env python3
"""
方案A — 64³ DNS-OOD 评估: 64³ V5 8-seed vs 32³ V5 基线
======================================================
用新生成的 64³ OOD 测试集 (eval_data/v5_ood_64_*.pt) 评估:
  - 8 个 64³ V5 seed 模型 (checkpoints/v5_64_seed{S}_best.pt)
  - 32³ V5 基线 (checkpoints/fno_v5_32_best.pt)   [对比: 32³模型吃64³输入]
  - 可选 V3/V4
输出: results/eval_chain64/v5_64_ood64_*.json + summary

用法 (每GPU一个模型):
  python3 eval_v5_64_ood.py --tag seed42 --seed 42 --gpu_id 0
  python3 eval_v5_64_ood.py --tag v5_32 --ckpt checkpoints/fno_v5_32_best.pt --gpu_id 0
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

BASE = '/mnt/mooncake-data/nsfno'
sys.path.insert(0, BASE)
from train_fno_v3 import VorticityFNO3D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    return VorticityFNO3D(in_c=3, out_c=3, width=64, modes_h=12, modes_w=12,
                          modes_d=8, n_layers=6).to(DEVICE)


def load_model(ckpt, fallback_stats_json=None):
    m = build_model()
    c = torch.load(ckpt, map_location=DEVICE)
    sd = c.get('model_state_dict', c)
    m.load_state_dict(sd); m.eval()
    stats = c.get('norm_stats')
    if not stats and fallback_stats_json and os.path.exists(fallback_stats_json):
        with open(fallback_stats_json) as f:
            stats = json.load(f)
    if not stats:
        raise RuntimeError(f"No norm_stats in {ckpt}")
    return m, stats


@torch.no_grad()
def predict(model, omega, stats, bs=8):
    out = []
    im, isv, tm, ts = stats['in_mean'], stats['in_std'], stats['tg_mean'], stats['tg_std']
    for i in range(0, omega.shape[0], bs):
        b = omega[i:i + bs].to(DEVICE)
        p = model((b - im) / isv)
        out.append((p * ts + tm).cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--eval_data', type=str, default=f'{BASE}/eval_data')
    ap.add_argument('--out_dir', type=str, default=f'{BASE}/results/eval_chain64')
    args = ap.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.manual_seed(0); np.random.seed(0)

    if args.seed is not None:
        ckpt = os.path.join(BASE, 'checkpoints', f'v5_64_seed{args.seed}_best.pt')
    elif args.ckpt:
        ckpt = os.path.join(BASE, args.ckpt)
    else:
        raise SystemExit("need --seed or --ckpt")
    if not os.path.exists(ckpt):
        print(f"SKIP {args.tag}: missing {ckpt}", flush=True)
        return
    try:
        model, stats = load_model(ckpt, f'{BASE}/data/v564_stats.json')
    except Exception as e:
        print(f"LOAD_FAILED {args.tag}: {e}", flush=True)
        return
    print(f"[{args.tag}] device {DEVICE} stats in_std={stats['in_std']:.4f}", flush=True)

    out = {'tag': args.tag}
    # 64³ time OOD
    t_ood = torch.load(os.path.join(args.eval_data, 'v5_ood_64_time.pt'), map_location='cpu')
    for dp, (X, Y) in t_ood.items():
        pred = predict(model, X, stats)
        out[f'time_dt_{dp}'] = {
            'rel_l2': float((pred - Y).norm().item() / (Y.norm().item() + 1e-8)),
            'n': int(X.shape[0]),
            'per_sample_rel_l2_mean': float(
                ((pred.flatten(1).norm(1)) / (Y.flatten(1).norm(1) + 1e-8)).mean().item()),
        }
        print(f"  time dt={dp}: rel_L2={out[f'time_dt_{dp}']['rel_l2']:.4f} (n={X.shape[0]})", flush=True)
    # 64³ intensity OOD
    i_ood = torch.load(os.path.join(args.eval_data, 'v5_ood_64_intensity.pt'), map_location='cpu')
    for et, (X, Y) in i_ood.items():
        if X.shape[0] == 0:
            continue
        pred = predict(model, X, stats)
        out[f'intensity_E{et}'] = {
            'rel_l2': float((pred - Y).norm().item() / (Y.norm().item() + 1e-8)),
            'n': int(X.shape[0]),
            'per_sample_rel_l2_mean': float(
                ((pred.flatten(1).norm(1)) / (Y.flatten(1).norm(1) + 1e-8)).mean().item()),
        }
        print(f"  intensity E={et}: rel_L2={out[f'intensity_E{et}']['rel_l2']:.4f} (n={X.shape[0]})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f'ood64_{args.tag}.json')
    with open(outp, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"Saved {outp}", flush=True)
    print("OK", flush=True)


if __name__ == '__main__':
    main()
