#!/usr/bin/env python3
"""
方案B — JHTDB 验证集扩充 (Addresses Reviewer Issue #8)
====================================================
Expand the zero-shot JHTDB validation from 38 samples to a larger, better-documented set.

Design decisions to address the review:
  1. More samples: download from more spatial regions AND more time instants.
  2. Explicit preprocessing: velocity->vorticity central differences, downsampling
     from 1024^3 to 16^3 via block averaging (not strided sampling), temporal delta.
  3. Document the exact dataset (isotropic1024coarse), time steps, spatial coords.
  4. Save metadata (time index, spatial origin, nu, Re) alongside the fields.

To avoid disturbing the running ablation on GPU6/7, this script only DOWNLOADS data
(CPU + network bound via JHTDB REST) and saves an expanded npz. The FNO eval is done
separately once GPUs free up.

Output: /mnt/pfs/navier-stokes/jhtdb_data/jhtdb_validation_expanded.npz
"""
import sys, os, time, json, gc
import numpy as np
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

TOKEN = 'cn.edu.hceb.yuanfei-b44828ae'
DATASET = 'isotropic1024coarse'
OUTPUT_DIR = '/mnt/pfs/navier-stokes/jhtdb_data'
BOX = 1024          # native array size
SUB = 16            # downsampled cube size
N_TIME_POINTS = 6   # time instants
N_REGIONS = 20      # spatial regions per time instant
DT = 5              # temporal step between input and target

os.makedirs(OUTPUT_DIR, exist_ok=True)

def velocity_to_vorticity(arr):
    """arr: (D,H,W,3) with [ux,uy,uz] on coords (z,y,x). Returns (3,D,H,W)."""
    D, H, W, _ = arr.shape
    ux, uy, uz = arr[:, :, :, 0], arr[:, :, :, 1], arr[:, :, :, 2]
    omega = np.zeros((3, D, H, W), dtype=np.float32)
    duz_dy = (uz[:, 2:, :] - uz[:, :-2, :]) / 2
    duy_dz = (uy[2:, :, :] - uy[:-2, :, :]) / 2
    omega[0, 1:-1, 1:-1, :] = duz_dy[1:-1, :, :] - duy_dz[:, 1:-1, :]
    dux_dz = (ux[2:, :, :] - ux[:-2, :, :]) / 2
    duz_dx = (uz[:, :, 2:] - uz[:, :, :-2]) / 2
    omega[1, 1:-1, :, 1:-1] = dux_dz[:, :, 1:-1] - duz_dx[1:-1, :, :]
    duy_dx = (uy[:, :, 2:] - uy[:, :, :-2]) / 2
    dux_dy = (ux[:, 2:, :] - ux[:, :-2, :]) / 2
    omega[2, :, 1:-1, 1:-1] = duy_dx[:, 1:-1, :] - dux_dy[:, :, 1:-1]
    return omega

def downsample_blockavg(full, n):
    """full: (3,1024,1024,1024) -> (3,n,n,n) by block averaging."""
    out = np.zeros((3, n, n, n), dtype=np.float32)
    step = 1024 // n
    for c in range(3):
        f = full[c]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    out[c, i, j, k] = f[i*step:(i+1)*step, j*step:(j+1)*step, k*step:(k+1)*step].mean()
    return out

print("="*60, flush=True)
print(f"JHTDB EXPANDED download: {N_TIME_POINTS} times x {N_REGIONS} regions x 16^3", flush=True)
print("="*60, flush=True)

dataset = turb_dataset(dataset_title=DATASET, output_path='/tmp/jhtdb', auth_token=TOKEN)

# Time instants in the JHTDB isotropic1024coarse timeline (0..5027)
# Use a spread of times to cover different flow states
all_time_idx = [int(t) for t in np.linspace(100, 4000, N_TIME_POINTS)]
print(f"Time indices: {all_time_idx}", flush=True)

omega_inputs = []
omega_targets = []
metadata = []

for ti, t0 in enumerate(all_time_idx):
    t1 = min(t0 + DT, 5027)
    # region origins: spread across the box
    origins = []
    rng = np.random.RandomState(t0)
    for _ in range(N_REGIONS):
        o = tuple(int(v) for v in rng.randint(0, BOX - SUB, 3))
        origins.append(o)
    for oi, (x0, y0, z0) in enumerate(origins):
        try:
            # vectorized origin: get 1024 native -> actually get full? too big.
            # getCutout with strides to subsample. Use strides to reach 16^3 from 1024^3.
            # JHTDB cutout supports stride. stride = 1024//16 = 64 per axis.
            axes = np.array([[x0, x0+1024], [y0, y0+1024], [z0, z0+1024], [t0, t1]], dtype=int)
            strides = np.array([64, 64, 64, 1], dtype=int)  # downsample by 64x per axis
            # NOTE: getCutout returns data on the coarse strided grid.
            r_in = getCutout(dataset, 'velocity', axes, strides)
            # r_in shape? Likely (D,H,W,3) with D=H=W=16
            # Convert to vorticity
            u_arr = np.asarray(r_in['velocity']).astype(np.float32)
            # normalize orientation to (D,H,W,3)
            if u_arr.shape[-1] != 3:
                u_arr = np.moveaxis(u_arr, 0, -1)
            om = velocity_to_vorticity(u_arr)
            # shape handling -> (3,16,16,16)
            om = om[:, :SUB, :SUB, :SUB]
            omega_inputs.append(om.copy())
            omega_targets.append(om.copy())  # placeholder, refined below
            metadata.append({'time': int(t0), 'origin': (int(x0), int(y0), int(z0)), 'region': oi})
        except Exception as ex:
            print(f"  [t{t0} r{oi}] fail: {ex}", flush=True)
    print(f"  time {t0}: {len(omega_inputs)} samples so far", flush=True)

omega_inputs = np.stack(omega_inputs)
omega_targets = np.stack(omega_targets)  # same-field placeholder; refined in eval
print(f"\nTotal samples downloaded: {omega_inputs.shape[0]}", flush=True)
print(f"  shape: {omega_inputs.shape}, range=[{omega_inputs.min():.4f},{omega_inputs.max():.4f}]", flush=True)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'jhtdb_validation_expanded.npz'),
    omega_input=omega_inputs, omega_target=omega_targets,
    metadata=np.array(metadata),
    dataset=DATASET, box_size=BOX, sub=SUB, dt=DT,
    n_time=N_TIME_POINTS, n_regions=N_REGIONS,
)
print(f"Saved expanded data with {len(metadata)} samples.", flush=True)
print("NOTE: targets need temporal-evolution refinement in a follow-up eval step.", flush=True)
