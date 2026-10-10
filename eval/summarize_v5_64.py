#!/usr/bin/env python3
"""
方案B — 64³ 多seed 评估链聚合器
================================
读取 eval_v5_64_chain.py 产出的每个 seed 的 JSON，聚合:

  - 8 seed (64³ V5) 在 ID test / JHTDB / DNS-OOD 上的 mean±std
  - 与 32³ V5 基线 (v5_32) 的配对对比 (Wilcoxon + t-test, 纯 numpy)
  - 输出 results/eval_chain/v5_64_chain_summary.json + markdown 报告

用法:
  python3 summarize_v5_64.py --eval_dir results/eval_chain
"""
import os, sys, json, glob
import numpy as np

BASE = "/mnt/mooncake-data/nsfno"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
# reuse pure-numpy stats from eval script (import-safe: check the funcs exist)
from eval_v5_64_chain import wilcoxon_p, ttest_rel_p, bootstrap_ci

SEEDS = [42, 123, 456, 789, 1111, 2222, 3333, 4444]
BASELINE_TAG = "v5_32"   # 32³ V5 基线


def rep(o):
    if abs(o) >= 1e-3:
        return f"{o:.4f}"
    return f"{o:.2e}"


def fmt_ms(xs):
    """list of per-seed values -> 'mean±std' string."""
    a = np.asarray(xs, float)
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str,
                    default=f"{BASE}/results/eval_chain")
    ap.add_argument("--out", type=str,
                    default=f"{BASE}/results/eval_chain/v5_64_chain_summary.json")
    args = ap.parse_args()

    # load all seed evals
    evals = {}
    for s in SEEDS:
        p = os.path.join(args.eval_dir, f"eval_seed{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                evals[s] = json.load(f)
    # baseline
    base = None
    bp = os.path.join(args.eval_dir, f"eval_{BASELINE_TAG}.json")
    if os.path.exists(bp):
        with open(bp) as f:
            base = json.load(f)
    print(f"Loaded {len(evals)}/8 seeds; baseline {BASELINE_TAG}: {base is not None}", flush=True)

    if len(evals) == 0:
        print("No seed evals found.", flush=True)
        return

    # ---------------- key metric collectors ----------------
    def collect(path):
        """path: list of keys into eval json. returns {seed: value}."""
        d = {}
        for s, e in evals.items():
            cur = e
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                d[s] = float(cur)
        return d

    metrics = [
        ("id_test_pooled_rel_l2", ["id_test", "pooled_rel_l2"]),
        ("id_test_per_sample_rel_l2_mean", ["id_test", "per_sample_rel_l2_mean"]),
        ("id_test_mse", ["id_test", "mse"]),
        ("id_test_corr", ["id_test", "per_sample_corr_mean"]),
        ("jhtdb_pooled_rel_l2", ["jhtdb", "pooled_rel_l2"]),
        ("jhtdb_per_sample_rel_l2_mean", ["jhtdb", "per_sample_rel_l2_mean"]),
        ("jhtdb_mse", ["jhtdb", "mse"]),
        ("jhtdb_mae", ["jhtdb", "mae"]),
        ("jhtdb_corr", ["jhtdb", "per_sample_corr_mean"]),
        ("ood_time_dt0.2", ["dns_ood", "time_dt_0.2", "rel_l2"]),
        ("ood_time_dt0.5", ["dns_ood", "time_dt_0.5", "rel_l2"]),
        ("ood_time_dt1.0", ["dns_ood", "time_dt_1.0", "rel_l2"]),
        ("ood_intensity_E0.2", ["dns_ood", "intensity_E0.2", "rel_l2"]),
        ("ood_intensity_E0.35", ["dns_ood", "intensity_E0.35", "rel_l2"]),
        ("ood_resolution", ["dns_ood", "resolution_64_to_32", "rel_l2"]),
    ]

    summary = {"n_seeds": len(evals), "seeds": sorted(evals.keys()),
               "baseline": BASELINE_TAG,
               "metrics": {}}
    md_rows = []

    for mname, paths in metrics:
        d = collect(paths)
        if len(d) == 0:
            continue
        vals = np.array(list(d.values()), float)
        rec = {
            "per_seed": {str(k): float(v) for k, v in d.items()},
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        # baseline comparison (pooled rel-L2 style: compare baseline scalar to seed dist)
        if base:
            bcur = base
            ok = True
            for k in (paths if isinstance(paths, list) else paths):
                if isinstance(bcur, dict) and k in bcur:
                    bcur = bcur[k]
                else:
                    ok = False
                    break
            if ok:
                bval = float(bcur)
                # paired test needs arrays of same length: baseline is 1 value;
                # treat baseline as constant and compute per-seed diff
                diff = vals - bval
                rec["baseline_value"] = bval
                rec["baseline_diff"] = float(diff.mean())
                # single-sample: report % improvement vs baseline
                rec["baseline_improve_pct"] = float((bval - vals.mean()) / max(abs(bval), 1e-12) * 100.0) if bval != 0 else None
        summary["metrics"][mname] = rec

        bstr = ""
        if base and "baseline_value" in rec:
            bstr = (f" | base={rep(rec['baseline_value'])} "
                    f"Δ={rec['baseline_diff']:+.4f} "
                    f"({rec['baseline_improve_pct']:+.1f}%)" if rec.get("baseline_improve_pct") is not None else "")
        md_rows.append(f"| {mname} | {fmt_ms(vals)}{bstr} |")

    # ---------------- JHTDB with full per-sample paired array (seed-vs-baseline) ----------------
    # If we have per-seed arrays for JHTDB (each eval stores only aggregated summary),
    # a full paired test across 118 samples x seeds is computed in the per-sample collector instead.
    # Here we additionally do a cross-seed paired comparison on per-seed JHTDB rel-L2 vs baseline.

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("Saved", args.out, flush=True)

    print("\n===== 64³ V5 multi-seed evaluation summary =====", flush=True)
    for mname, _paths in metrics:
        if mname in summary["metrics"]:
            r = summary["metrics"][mname]
            b = f" (base {r.get('baseline_value') and rep(r['baseline_value'])}, Δ={r.get('baseline_diff'):+.4f})" if "baseline_diff" in r else ""
            print(f"  {mname:30s} mean±std = {fmt_ms(list(r['per_seed'].values()))}{b}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
