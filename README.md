# NS-FNO

**Fourier Neural Operator for 3D Vorticity Evolution with Multi-Physics Regularity Constraints**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

Official code for the manuscript *"NS-FNO: Fourier Neural Operator for
Three-Dimensional Vorticity Evolution with Multi-Physics Regularity
Constraints"* (submitted to the **Journal of Computational Physics**).

NS-FNO is a **56.67M-parameter three-dimensional Fourier Neural Operator** that
predicts vorticity-field evolution in incompressible Navier–Stokes flows. It is
trained on a **genuine DNS corpus** (not hand-crafted synthetic fields) and
employs a composite loss jointly encoding **six physics-aware regularity
criteria**: the Beale–Kato–Majda (BKM) blow-up criterion, the Prodi–Serrin
condition, the divergence-free constraint, enstrophy conservation,
vortex-stretching dynamics, and viscous dissipation.

---

## Key results (manuscript v5.2)

| Result | Value |
|---|---|
| Validation loss, synthetic baseline (V3) | 0.02267 |
| Validation loss, genuine-DNS training (V5) | 0.000058 (**≈390× improvement**) |
| JHTDB pooled relative-L² (118 real samples, V5) | 0.672 |
| JHTDB correlation (V5) | 0.792, *p* < 10⁻²⁰ (paired tests) |
| OOD generalization vs. synthetic baselines | 1.4–2.4× (temporal), 1.4–1.8× (intensity), 1.6–1.9× (resolution) |
| Physics-loss ablation (7 configs, ID) | 0.0093 ± 0.00005 (neutral regularizers) |
| BKM risk-indicator separation (synthetic contract-vs-decay) | 11.4× (discrimination, not singularity claim) |
| Resolution independence | 64³ over 8 seeds matches 32³ within 0.005; mixed 32³+64³ training −12% ID, no OOD regression |
| Training budget (V5, RTX 4090D) | ≈ 2.1 h |
| Inference cost | ≈ 0.8 ms / sample |

These numbers are verified against the raw eval JSONs in [`results/`](results/).

---

## Repository layout

```
├── dns/                        # Pseudo-spectral DNS solver & corpus generation
│   ├── dns_solver.py           #   CPU reference solver (spectral, RK3, 3/2 dealiasing)
│   ├── dns_solver_torch.py     #   GPU solver used to generate the corpus
│   ├── dns_ns3d.py             #   NS3D-style helpers / verification
│   ├── gen_dns_corpus.py       #   Generate the 104-trajectory × 64³ genuine-DNS corpus
│   └── gen_dns_corpus_v2.py    #   Alternative corpus generator (ablation studies)
├── jhtdb/                      # Real-turbulence (JHTDB isotropic1024coarse) pipeline
│   ├── jhtdb_download_expanded.py  # Download 6 timesteps × 20 regions
│   ├── jhtdb_expand.py             # 1024³ → 16³ block-averaged cutouts (80 expanded)
│   ├── jhtdb_expanded_eval.py      # Evaluate FNO on the 118-sample JHTDB benchmark
│   └── jhtdb_stats.py              # Per-sample statistics + bootstrap CIs
├── src/
│   ├── vorticity_analyzer.py   # VorticityFNO3D architecture + EnhancedPhysicsLoss (all six criteria)
│   ├── train_fno_v5_ref.py     # Reference training (32³ & 64³, multi-seed)
│   └── train_fno_v5_mix.py     # Mixed-resolution (32³+64³) training
├── eval/
│   ├── unified_eval.py         # In-distribution + JHTDB evaluation
│   ├── eval_v5_64_ood.py       # 64³-grid OOD evaluation
│   ├── eval_v5_64_chain.py     # Multi-step rollout chain
│   ├── ablation_completion.py  # Complete the 7-config physics-loss ablation
│   └── ablation_physics_eval.py# Ablation evaluation
├── results/                    # Raw eval JSONs (multi-seed) + PAPER_TABLES.md
├── figures/                    # Manuscript figures
└── paper/                      # Manuscript source (.tex) and compiled PDF
```

---

## Data

- **Genuine-DNS corpus:** generated with the included pseudo-spectral solver
  (64³, third-order low-storage Runge–Kutta, 3/2 zero-padding dealiasing,
  spectral divergence-free projection, energy-conserving in the inviscid limit,
  *Re* = 4000-equivalent viscosity ν = 0.01). 104 trajectories with random ICs
  sampled from a Gaussian spectral envelope (k₀ = 2.5) yield **6,136 real
  trajectory pairs** (snapshot Δt = 0.1, trajectory length T = 6.0). Regenerate
  with `dns/gen_dns_corpus.py`; exact seeds are fixed for reproducibility.
- **JHTDB real turbulence:** 118 evaluation samples cut from
  `isotropic1024coarse` (*Re*_λ ≈ 433), 1024³ → 16³ block-averaged, drawn
  independently (38 original + 80 expanded from 6 timesteps × 20 regions).
  Download via `jhtdb/jhtdb_download_expanded.py`.

The full generated corpus is archived on Zenodo (DOI reserved at publication);
the evaluation cutouts and all scripts are included here.

---

## Installation & quick start

```bash
conda create -n nsfno python=3.11 -y && conda activate nsfno
pip install -r requirements.txt          # torch>=2.0, numpy, matplotlib, scipy, tqdm
```

Verify the DNS solver against the Taylor–Green analytic decay:

```bash
python dns/dns_solver_torch.py --self-test
```

Generate the genuine-DNS corpus (64³, 104 trajectories; ~hours on one GPU):

```bash
python dns/gen_dns_corpus.py --out ./corpus --n-trajs 104 --t-end 6.0
```

Train the reference V5 model (32³, single seed for quick check; 8 seeds for the paper):

```bash
python src/train_fno_v5_ref.py --data ./corpus --out ./ckpt/v5_32 --seed 123
```

Evaluate on the JHTDB benchmark:

```bash
python jhtdb/jhtdb_expanded_eval.py --ckpt ./ckpt/v5_32 --out ./results
```

---

## Citation

```bibtex
@article{nsfno2026,
  title   = {NS-FNO: Fourier Neural Operator for Three-Dimensional Vorticity
             Evolution with Multi-Physics Regularity Constraints},
  author  = {Yuan, F. and Liu, X. and Wang, G.},
  journal = {Journal of Computational Physics},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

MIT — see [LICENSE](LICENSE). The JHTDB data are subject to the
[JHTDB terms of use](http://turbulence.pha.jhu.edu/).
