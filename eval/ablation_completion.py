"""
方案B — 补齐缺失消融配置 (MSE-only + nops) for 5 seeds
=====================================================
Reviewer Experiment #1: MSE-only NS-FNO vs multi-physics NS-FNO
(same architecture, same params, same training budget).

Configs:
  - mse  : all physics weights 0 -> pure MSE baseline (the critical ablation)
  - nops : Prodi-Serrin removed (weight 0)

Same 5 seeds as ablation_multi_seed. Saves to results/ablation_mse_completion_results.json
with the SAME schema as ablation_multi_seed_results.json so results can be merged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, json, sys, threading
from datetime import datetime

sys.path.insert(0, '/mnt/pfs/navier-stokes')
from train_fno_v3 import generate_data, VorticityFNO3D
from vorticity_analyzer import EnhancedPhysicsLoss

os.environ['PYTHONUNBUFFERED'] = '1'

CONFIGS = {
    'mse':  {'div': 0.0, 'bkm': 0.0, 'smooth': 0.0, 'enstrophy': 0.0, 'stretch': 0.0, 'ps': 0.0},
    'nops': {'div': 1.0, 'bkm': 0.5, 'smooth': 0.1, 'enstrophy': 0.1, 'stretch': 0.1, 'ps': 0.0},
}
SEEDS = [42, 123, 456, 789, 1111]
EPOCHS = 50
BATCH_SIZE = 64
LR = 3e-4

CONFIG_NAMES = list(CONFIGS.keys())
all_results = []
result_lock = threading.Lock()
log_lock = threading.Lock()

def log(msg):
    with log_lock:
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {msg}", flush=True)

def compute_velocity_from_vorticity(omega):
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
    psi_x, psi_y, psi_z = psi_hat[:, 0:1], psi_hat[:, 1:2], psi_hat[:, 2:3]
    ux_hat = -(1j * ky_c * psi_z - 1j * kz_c * psi_y)
    uy_hat = -(1j * kz_c * psi_x - 1j * kx_c * psi_z)
    uz_hat = -(1j * kx_c * psi_y - 1j * ky_c * psi_x)
    ux = torch.fft.irfftn(ux_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uy = torch.fft.irfftn(uy_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    uz = torch.fft.irfftn(uz_hat, s=[H, W, D], dim=[2, 3, 4]).permute(0, 1, 4, 2, 3)
    return torch.cat([ux, uy, uz], dim=1)

def save_results():
    with result_lock:
        results_dir = '/mnt/pfs/navier-stokes/results'
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'ablation_mse_completion_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

# Reuse existing ablation results if present to avoid re-running completed configs
def load_existing():
    p = '/mnt/pfs/navier-stokes/results/ablation_multi_seed_results.json'
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []

def run_config(config_name, config_weights, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    local_results = []
    for seed_idx, seed in enumerate(SEEDS):
        log(f"[GPU{gpu_id}][{config_name}] Seed {seed} ({seed_idx+1}/{len(SEEDS)})...")
        all_in, all_tg, stats = generate_data(seed=seed)
        n_val = max(1, int(len(all_in) * 0.12))
        train_ds = torch.utils.data.TensorDataset(all_in[:-n_val], all_tg[:-n_val])
        val_ds = torch.utils.data.TensorDataset(all_in[-n_val:], all_tg[-n_val:])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        torch.manual_seed(seed); np.random.seed(seed)
        model = VorticityFNO3D(in_c=3, out_c=3, width=64, modes_h=12, modes_w=12, modes_d=8, n_layers=6).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        physics_loss = EnhancedPhysicsLoss(
            viscosity=0.01,
            lambda_div=config_weights['div'], lambda_bkm=config_weights['bkm'],
            lambda_smooth=config_weights['smooth'], lambda_enstrophy=config_weights['enstrophy'],
            lambda_ps=config_weights['ps'],).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        best_val = float('inf'); best_epoch = -1; t0 = time.time()

        for epoch in range(EPOCHS):
            model.train(); tl = 0.0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x)
                loss_mse = F.mse_loss(pred, y)
                total_weight = sum(config_weights.values())
                if total_weight > 0:
                    vel_pred = compute_velocity_from_vorticity(pred)
                    _, ld = physics_loss(vel_pred, pred)
                    loss_phys = 0.0
                    map_d = {'div':'divergence','bkm':'bkm','smooth':'smoothness','enstrophy':'enstrophy','stretch':'stretching','ps':'prodi_serrin'}
                    for k, w in config_weights.items():
                        if w > 0:
                            val = ld.get(map_d[k], 0.0)
                            if not isinstance(val, torch.Tensor):
                                val = torch.tensor(val, device=device)
                            loss_phys = loss_phys + val * w
                    loss = loss_mse + loss_phys
                else:
                    loss = loss_mse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item()
            avg_tl = tl / len(train_loader)
            model.eval(); vl = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    vl += F.mse_loss(model(x), y).item()
            avg_vl = vl / len(val_loader)
            scheduler.step()
            if (epoch+1) % 10 == 0 or epoch == 0:
                log(f"[GPU{gpu_id}][{config_name}] Seed {seed} Ep {epoch+1:3d}/{EPOCHS}: train={avg_tl:.6f}, val={avg_vl:.6f} ({time.time()-t0:.0f}s)")
            if avg_vl < best_val: best_val = avg_vl; best_epoch = epoch
        log(f"[GPU{gpu_id}][{config_name}] Seed {seed} DONE best_val={best_val:.6f} @ep{best_epoch+1} {time.time()-t0:.0f}s")
        local_results.append({'config': config_name, 'seed': seed, 'best_val': best_val,
                              'best_epoch': best_epoch, 'total_time': time.time()-t0, 'n_params': n_params})
        del model, physics_loss, train_loader, val_loader, all_in, all_tg
        torch.cuda.empty_cache()
        save_results()
    return local_results

def run_and_collect(config_name, gpu_id):
    local = run_config(config_name, CONFIGS[config_name], gpu_id)
    with result_lock:
        all_results.extend(local)
        save_results()

def main():
    print(f"\n{'='*60}\n  Completion: MSE-only + nops (5 seeds each)\n  GPUs: {torch.cuda.device_count()}\n{'='*60}\n", flush=True)
    # load existing to merge
    global all_results
    all_results = load_existing()
    t_start = time.time()
    threads = []
    for i, cname in enumerate(CONFIG_NAMES):
        t = threading.Thread(target=run_and_collect, args=(cname, i))
        threads.append(t); t.start(); log(f"Launched {cname} on GPU{i}")
    for t in threads: t.join()
    print(f"完成! 总耗时 { (time.time()-t_start)/60:.1f} min", flush=True)
    save_results()
    print(f"Saved. Total runs now: {len(all_results)}", flush=True)

if __name__ == '__main__':
    main()
