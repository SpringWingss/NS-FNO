#!/usr/bin/env python3
"""
方案B — 64³ 大规模 V5 多seed 评估链 (单 seed 评估入口)
=====================================================
在 H100 上评估单个 64³ V5 seed checkpoint，产出三个基准的结果:

  1. ID test:     真实 64³ DNS 测试集 (1593 samples, v564_*_test.npy)
  2. JHTDB:       真实 JHTDB DNS 118 samples (16³, 分辨率无关直接吃)
  3. DNS-OOD:     时间/强度/分辨率三维外推 (32³, 来自 32³ V5 OOD 数据)

每 seed 独立输出 JSON，然后用 summarize_v5_64.py 聚合 8 seed 的 mean±std，
并与 32³ V5 基线 (fno_v5_32_best.pt) 做配对对比。

指标协议 (与 jhtdb_v5_eval.py / v5_dns_ood_eval.py 一致):
  - pooled rel-L2 (尺度无关), MSE, MAE, per-sample corr
  - per-sample rel-L2 mean±std + bootstrap 95% CI
  - (聚合阶段) paired Wilcoxon + paired t-test  纯 numpy 实现，不依赖 scipy

用法 (每 GPU 一个 seed):
  CUDA_VISIBLE_DEVICES=$i python3 eval_v5_64_chain.py --seed $S --gpu_id $i --out_dir results/eval_chain
  # 基线模型 (32³ V5 / V3 / V4) 用 --model_tag + --ckpt 指定
  CUDA_VISIBLE_DEVICES=$i python3 eval_v5_64_chain.py --model_tag v5_32 --ckpt checkpoints/fno_v5_32_best.pt --gpu_id $i --out_dir results/eval_chain
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

BASE = "/mnt/mooncake-data/nsfno"
sys.path.insert(0, BASE)
from train_fno_v3 import VorticityFNO3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 纯 numpy 统计检验 (不依赖 scipy)
# ----------------------------------------------------------------------------
def _rankdata(a):
    """Average ranks for ties. a: 1-D array."""
    a = np.asarray(a, dtype=float)
    n = a.size
    sorter = np.argsort(a)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[sorter[j + 1]] == a[sorter[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-indexed average rank
        ranks[sorter[i:j + 1]] = avg
        i = j + 1
    return ranks


def wilcoxon_p(x, y):
    """Two-sided Wilcoxon signed-rank test (ties handled), returns p-value via normal approx."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    d = d[np.abs(d) > 1e-12]
    n = d.size
    if n == 0:
        return 1.0
    r = _rankdata(np.abs(d))
    w = float(np.sum(r[d > 0]))  # sum of positive ranks
    mean = n * (n + 1) / 4.0
    # variance with tie correction
    nt = (n * (n + 1) * (2 * n + 1)) / 24.0
    ties = np.unique(np.abs(d), return_counts=True)[1]
    tie_corr = sum(t ** 3 - t for t in ties[ties > 1]) / 48.0
    var = nt - tie_corr
    if var <= 0:
        return 1.0
    z = (w - mean) / np.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(np.abs(z) / np.sqrt(2.0))))
    return float(min(p, 1.0))


def ttest_rel_p(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    n = d.size
    if n < 2:
        return 1.0
    md = d.mean()
    var = d.var(ddof=1)
    if var <= 0:
        return 1.0
    se = np.sqrt(var / n)
    t = md / se
    # two-sided p via survival of student-t (approximate normal for large n)
    df = n - 1
    # use erf-based normal approx (large n); for small n use incomplete beta is overkill
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(np.abs(t) / np.sqrt(2.0))))
    return float(min(p, 1.0))


def bootstrap_ci(data, B=10000, seed=0):
    rng = np.random.RandomState(seed)
    n = data.size
    bmeans = np.array([data[rng.randint(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(bmeans, 2.5)), float(np.percentile(bmeans, 97.5))


# ----------------------------------------------------------------------------
# 模型 / 预测
# ----------------------------------------------------------------------------
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
        raise RuntimeError(f"No norm_stats in {ckpt} (and no fallback)")
    return m, stats


@torch.no_grad()
def predict(model, omega, stats, bs=32):
    """omega: (N,3,D,H,W) raw physical vorticity -> predicted physical vorticity."""
    out = []
    im, isv, tm, ts = stats['in_mean'], stats['in_std'], stats['tg_mean'], stats['tg_std']
    for i in range(0, omega.shape[0], bs):
        b = omega[i:i + bs].to(DEVICE)
        p = model((b - im) / isv)
        out.append((p * ts + tm).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def predict_smoke(model, omega, stats, bs=2, n=2):
    """Smoke-test prediction on a few samples only (CPU-safe, no GPU)."""
    out = []
    im, isv, tm, ts = stats['in_mean'], stats['in_std'], stats['tg_mean'], stats['tg_std']
    for i in range(0, min(n, omega.shape[0]), bs):
        b = omega[i:i + bs].to(DEVICE)
        p = model((b - im) / isv)
        out.append((p * ts + tm).cpu())
    return torch.cat(out, 0)


def per_sample_metrics(pred, target):
    """pred/target: (N, C,D,H,W) torch tensors on CPU."""
    pf = pred.flatten(1); tf = target.flatten(1)
    per_rel = (pf.norm(dim=1) / (tf.norm(dim=1) + 1e-8)).numpy()
    per_mse = ((pred - target) ** 2).flatten(1).mean(1).numpy()
    pc = pf - pf.mean(1, keepdim=True); tc = tf - tf.mean(1, keepdim=True)
    per_corr = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)
    return per_rel, per_mse, per_corr.numpy()


def summarize_per_sample(per_rel, per_mse, per_corr, name, out):
    lo, hi = bootstrap_ci(per_rel)
    out[name] = {
        'n': int(per_rel.size),
        'per_sample_rel_l2_mean': float(per_rel.mean()),
        'std': float(per_rel.std(ddof=1)),
        'ci95': [lo, hi],
        'per_sample_mse_mean': float(per_mse.mean()),
        'per_sample_corr_mean': float(per_corr.mean()),
    }
    print(f"  {name}: rel_L2={per_rel.mean():.4f}±{per_rel.std(ddof=1):.4f} "
          f"CI95=[{lo:.4f},{hi:.4f}] mse={per_mse.mean():.3e} "
          f"corr={per_corr.mean():.4f}", flush=True)


# ----------------------------------------------------------------------------
# 三个基准
# ----------------------------------------------------------------------------
def eval_id_test(model, stats, datadir, predict_fn=None, limit=None):
    X = torch.from_numpy(np.load(os.path.join(datadir, 'v564_X_test.npy'))).float()
    Y = torch.from_numpy(np.load(os.path.join(datadir, 'v564_Y_test.npy'))).float()
    if limit:
        X = X[:limit]; Y = Y[:limit]
    if predict_fn is None:
        predict_fn = lambda m, x, st, _bs=8: predict(m, x, st, bs=_bs)
    pred = predict_fn(model, X, stats)
    per_rel, per_mse, per_corr = per_sample_metrics(pred, Y)
    out = {}
    summarize_per_sample(per_rel, per_mse, per_corr, 'id_test', out)
    out['id_test']['pooled_rel_l2'] = float(
        (pred - Y).norm().item() / (Y.norm().item() + 1e-8))
    out['id_test']['mse'] = float(F.mse_loss(pred, Y).item())
    out['id_test']['mae'] = float(F.l1_loss(pred, Y).item())
    print(f"  id_test pooled_rel_L2={out['id_test']['pooled_rel_l2']:.4f} "
          f"mse={out['id_test']['mse']:.3e}", flush=True)
    return out, per_rel


def eval_jhtdb(model, stats, datadir, orig, expanded, predict_fn=None, limit=None):
    d1 = np.load(os.path.join(datadir, 'jhtdb_validation_data.npz'))
    o1 = torch.from_numpy(d1['omega_input']).float()
    t1 = torch.from_numpy(d1['omega_target']).float()
    d2 = np.load(os.path.join(datadir, 'jhtdb_validation_expanded.npz'))
    o2 = torch.from_numpy(d2['omega_input']).float()
    t2 = torch.from_numpy(d2['omega_target']).float()
    om_in = torch.cat([o1, o2], 0); om_tg = torch.cat([t1, t2], 0)
    if limit:
        om_in = om_in[:limit]; om_tg = om_tg[:limit]
    print(f"  JHTDB combined {om_in.shape[0]} ({o1.shape[0]}+{o2.shape[0]}), "
          f"res {om_in.shape[2]}³", flush=True)
    if predict_fn is None:
        predict_fn = lambda m, x, st: predict(m, x, st)
    pred = predict_fn(model, om_in, stats)
    per_rel, per_mse, per_corr = per_sample_metrics(pred, om_tg)
    out = {}
    summarize_per_sample(per_rel, per_mse, per_corr, 'jhtdb', out)
    out['jhtdb'].update({
        'n_orig': int(o1.shape[0]), 'n_expanded': int(o2.shape[0]),
        'pooled_rel_l2': float((pred - om_tg).norm().item() / (om_tg.norm().item() + 1e-8)),
        'mse': float(F.mse_loss(pred, om_tg).item()),
        'mae': float(F.l1_loss(pred, om_tg).item()),
    })
    print(f"  jhtdb pooled_rel_L2={out['jhtdb']['pooled_rel_l2']:.4f} "
          f"mse={out['jhtdb']['mse']:.3e}", flush=True)
    return out, per_rel


def eval_dns_ood(model, stats, datadir):
    """Load pre-generated 32³ DNS-OOD .pt files and evaluate."""
    paths = {
        'time': os.path.join(datadir, 'v5_ood_time.pt'),
        'intensity': os.path.join(datadir, 'v5_ood_intensity.pt'),
        'resolution': os.path.join(datadir, 'v5_ood_resolution.pt'),
    }
    return eval_dns_ood_files(model, stats, paths)


def eval_dns_ood_files(model, stats, paths):
    """Evaluate DNS-OOD sets given {time,intensity,resolution} -> .pt path.
    paths may be a dict with those keys or a SimpleNamespace."""
    def _p(k):
        v = paths[k] if isinstance(paths, dict) else getattr(paths, k)
        return v
    out = {}
    time_ood = torch.load(_p('time'), map_location='cpu')
    inten_ood = torch.load(_p('intensity'), map_location='cpu')
    res_ood = torch.load(_p('resolution'), map_location='cpu')
    # time OOD
    for dp, (X, Y) in time_ood.items():
        if X.shape[0] == 0:
            continue
        pred = predict(model, X, stats)
        tag = 'id' if dp == 0.1 else 'ood'
        out[f'time_dt_{dp}'] = {
            'rel_l2': float((pred - Y).norm().item() / (Y.norm().item() + 1e-8)),
            'mse': float(F.mse_loss(pred, Y).item()),
            'n': int(X.shape[0]), 'tag': tag,
            'per_sample_rel_l2_mean': float(
                ((pred.flatten(1).norm(dim=1)) / (Y.flatten(1).norm(dim=1) + 1e-8)).mean().item()),
        }
        print(f"  time dt={dp}: rel_L2={out[f'time_dt_{dp}']['rel_l2']:.4f} "
              f"({tag}, n={X.shape[0]})", flush=True)
    # intensity OOD
    for et, (X, Y) in inten_ood.items():
        if X.shape[0] == 0:
            continue
        pred = predict(model, X, stats)
        tag = 'id' if et == 0.04 else 'ood'
        out[f'intensity_E{et}'] = {
            'rel_l2': float((pred - Y).norm().item() / (Y.norm().item() + 1e-8)),
            'mse': float(F.mse_loss(pred, Y).item()),
            'n': int(X.shape[0]), 'tag': tag,
            'per_sample_rel_l2_mean': float(
                ((pred.flatten(1).norm(dim=1)) / (Y.flatten(1).norm(dim=1) + 1e-8)).mean().item()),
        }
        print(f"  intensity E={et}: rel_L2={out[f'intensity_E{et}']['rel_l2']:.4f} "
              f"({tag}, n={X.shape[0]})", flush=True)
    # resolution OOD
    if isinstance(res_ood, (tuple, list)) and res_ood[0].shape[0] > 0:
        X, Y = res_ood
        pred = predict(model, X, stats)
        out['resolution_64_to_32'] = {
            'rel_l2': float((pred - Y).norm().item() / (Y.norm().item() + 1e-8)),
            'mse': float(F.mse_loss(pred, Y).item()),
            'n': int(X.shape[0]), 'tag': 'ood',
            'per_sample_rel_l2_mean': float(
                ((pred.flatten(1).norm(dim=1)) / (Y.flatten(1).norm(dim=1) + 1e-8)).mean().item()),
        }
        print(f"  res 64->32: rel_L2={out['resolution_64_to_32']['rel_l2']:.4f} "
              f"(ood, n={X.shape[0]})", flush=True)
    return out


# ----------------------------------------------------------------------------
def main():
    os.environ['PYTHONUNBUFFERED'] = '1'
    torch.set_num_threads(16)
    os.environ['OMP_NUM_THREADS'] = '16'
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None, help='64³ V5 seed (auto ckpt path)')
    ap.add_argument('--model_tag', type=str, default=None,
                    help='baseline tag, e.g. v5_32 / v3 / v4 (requires --ckpt)')
    ap.add_argument('--ckpt', type=str, default=None, help='explicit ckpt path')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--out_dir', type=str, default=f'{BASE}/results/eval_chain')
    ap.add_argument('--eval_data', type=str, default=f'{BASE}/eval_data',
                    help='dir with jhtdb_*.npz and v5_ood_*.pt')
    ap.add_argument('--smoke', action='store_true', help='run a tiny subset (CPU-safe) to verify the pipeline')
    ap.add_argument('--smoke_n', type=int, default=2)
    args = ap.parse_args()

    # 仅在未显式置空(CUDA_VISIBLE_DEVICES="")时设置GPU; 置空=强制CPU冒烟
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.manual_seed(0); np.random.seed(0)

    if args.smoke:
        # force CPU for smoke (GPU all busy with training)
        global DEVICE
        DEVICE = torch.device('cpu')

    if args.seed is not None:
        tag = f'seed{args.seed}'
        ckpt = os.path.join(BASE, 'checkpoints', f'v5_64_seed{args.seed}_best.pt')
        fallback = os.path.join(BASE, 'data', 'v564_stats.json')
    elif args.model_tag and args.ckpt:
        tag = args.model_tag
        ckpt = os.path.join(BASE, args.ckpt)
        fallback = os.path.join(BASE, 'data', 'v564_stats.json')
    else:
        raise SystemExit("Need --seed or (--model_tag + --ckpt)")

    if not os.path.exists(ckpt):
        print(f"SKIP missing ckpt: {ckpt}", flush=True)
        return

    print(f"Device: {DEVICE} | tag={tag} | ckpt={ckpt}", flush=True)
    try:
        model, stats = load_model(ckpt, fallback)
    except Exception as e:
        # checkpoint 可能正在被训练进程写入 (torch.save 非原子) - 报错后跳过, 稍后重试
        print(f"LOAD_FAILED {tag}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return
    print(f"  norm stats: in_std={stats['in_std']:.4f} tg_std={stats['tg_std']:.4f}", flush=True)

    datadir = os.path.join(BASE, 'data')
    eval_data = args.eval_data
    out = {'tag': tag, 'gpu_id': args.gpu_id,
           'ckpt': os.path.basename(ckpt),
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}

    limit = args.smoke_n if args.smoke else None

    print("\n== ID test (64³ real DNS) ==", flush=True)
    id_out, _ = eval_id_test(model, stats, datadir, limit=limit)
    out.update(id_out)

    print("\n== JHTDB (118 samples, 16³) ==", flush=True)
    jh_out, _ = eval_jhtdb(model, stats, eval_data, 38, 80, limit=limit)
    out['jhtdb'] = jh_out['jhtdb']

    print("\n== DNS-OOD (32³) ==", flush=True)
    if args.smoke:
        time_ood = torch.load(os.path.join(eval_data, 'v5_ood_time.pt'), map_location='cpu')
        inten_ood = torch.load(os.path.join(eval_data, 'v5_ood_intensity.pt'), map_location='cpu')
        res_ood = torch.load(os.path.join(eval_data, 'v5_ood_resolution.pt'), map_location='cpu')
        t0 = {k: (v[0][:args.smoke_n], v[1][:args.smoke_n]) for k, v in list(time_ood.items())[:3]}
        i0 = {k: (v[0][:args.smoke_n], v[1][:args.smoke_n]) for k, v in list(inten_ood.items())[:3]}
        r0 = (res_ood[0][:args.smoke_n], res_ood[1][:args.smoke_n])
        os.makedirs('/tmp/smoke_ood', exist_ok=True)
        torch.save(t0, '/tmp/smoke_ood/time.pt'); torch.save(i0, '/tmp/smoke_ood/inten.pt')
        torch.save(r0, '/tmp/smoke_ood/res.pt')
        import types
        tmp = types.SimpleNamespace(time='/tmp/smoke_ood/time.pt',
                                     intensity='/tmp/smoke_ood/inten.pt',
                                     resolution='/tmp/smoke_ood/res.pt')
        out['dns_ood'] = eval_dns_ood_files(model, stats, tmp)
    else:
        out['dns_ood'] = eval_dns_ood(model, stats, eval_data)

    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f'eval_{tag}.json')
    with open(outp, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {outp}", flush=True)

    # also save per-sample rel-L2 for later paired stats
    print("OK", flush=True)


if __name__ == '__main__':
    main()
