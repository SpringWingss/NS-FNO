#!/usr/bin/env python3
"""
方案B — 3D 伪谱 DNS 求解器 (torch GPU version)
==============================================
GPU-accelerated pseudo-spectral incompressible Navier-Stokes on [0,2pi)^3.
Validated against the numpy CPU version (same algorithm): RK3 + 3/2 dealiasing +
spectral div-free projection.

Shapes: uhat (3,N,N,N) complex128 torch tensor on CUDA.
All linear ops and FFTs are on GPU; only diagnostics reduce to CPU.

Modes:
  gen_tg_trajectory(...)  -> Taylor-Green decay trajectory (velocity + vorticity + energy)
  gen_isotropic(...)      -> (future) forced isotropic turbulence
  validation harness       -> TG per-mode analytic decay check

NOTE (numerics): complex128 is used to avoid aliasing/drift errors in long rolls.
For very large N use float32 carefully and monitor energy.
"""
import torch
import numpy as np
import time, json

TORCH_DTYPE = torch.complex128  # numerical safety for validation

def make_k(N, device):
    k = torch.fft.fftfreq(N, device=device) * N   # integer wavenumbers
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing='ij')
    K2 = KX*KX + KY*KY + KZ*KZ
    # dealias mask (2/3 rule)
    mask = (KX.abs() < 2*N/3) & (KY.abs() < 2*N/3) & (KZ.abs() < 2*N/3)
    return KX, KY, KZ, K2, mask

def project(uhat, KX, KY, KZ, K2, mask, apply_mask=True):
    """div-free projection: P = I - k k^T/k^2 (k=0 -> 0)."""
    K2s = K2.clone(); K2s[0,0,0] = 1.0
    kdotu = KX*uhat[0] + KY*uhat[1] + KZ*uhat[2]
    uhat = uhat - torch.stack([KX*kdotu, KY*kdotu, KZ*kdotu], 0) / K2s.unsqueeze(0)
    if apply_mask:
        uhat = uhat * mask.unsqueeze(0)
    uhat[:, 0,0,0] = 0.0
    return uhat

def nonlinear_rhs(uhat, N, dtype):
    """-P(u.grad u) via 3/2 dealiasing (torch).
    CRITICAL: pad-embedding + ifftn normalizes by Mp^3, giving physical field
    scaled by N^3/Mp^3. Multiply by (Mp/N)^3 to recover true physical values."""
    Mp = 3*N//2
    dev = uhat.device
    sc = (Mp**3) / (N**3)
    up = torch.zeros((3, Mp, Mp, Mp), dtype=dtype, device=dev)
    up[:, :N, :N, :N] = uhat
    kp = torch.fft.fftfreq(Mp, device=dev) * Mp
    KX, KY, KZ = torch.meshgrid(kp, kp, kp, indexing='ij')
    w = torch.fft.ifftn(up, dim=(1,2,3)) * sc
    gw0 = torch.fft.ifftn(1j*KX.unsqueeze(0)*up, dim=(1,2,3)) * sc
    gw1 = torch.fft.ifftn(1j*KY.unsqueeze(0)*up, dim=(1,2,3)) * sc
    gw2 = torch.fft.ifftn(1j*KZ.unsqueeze(0)*up, dim=(1,2,3)) * sc
    nlin = w[0:1]*gw0 + w[1:2]*gw1 + w[2:3]*gw2
    adv = torch.fft.fftn(nlin, dim=(1,2,3))[:, :N, :N, :N]
    return adv

def rhs(uhat, nu, K2, N, dtype, apply_mask=True, force=None, KX=None,KY=None,KZ=None,mask=None):
    adv = nonlinear_rhs(uhat, N, dtype)
    out = adv - nu*K2.unsqueeze(0)*uhat
    if force is not None:
        f = spectral_force(uhat, K2, N, force['kmin'], force['kmax'], force['P'])
        out = out + f
    # project after full rhs (on nonlinear only is standard; we project result)
    return out

def spectral_force(uhat, K, N, kmin, kmax, P_norm):
    """Deterministic forcing maintaining energy in shell [kmin,kmax) toward target."""
    shell = (K >= kmin) & (K < kmax)
    Es = (uhat.abs()**2)[:, shell].sum()
    if Es < 1e-30:
        return torch.zeros_like(uhat)
    eps = P_norm * (Es * (N**3))  # scaled injection
    f = torch.zeros_like(uhat)
    f[:, shell] = eps * uhat[:, shell] / (Es + 1e-30)
    return f

def rk3_step(uhat, dt, nu, K2, N, dtype, force=None, KX=None,KY=None,KZ=None,mask=None):
    def f(u):
        adv = nonlinear_rhs(u, N, dtype)
        adv = project(adv, KX, KY, KZ, K2, mask)   # project nonlinear ONLY (matches numpy)
        out = adv - nu*K2.unsqueeze(0)*u
        if force is not None:
            out = out + spectral_force(u, K2, N, force['kmin'], force['kmax'], force['P'])
        return out
    k1 = f(uhat)
    u1 = uhat + dt*0.5*k1
    k2 = f(u1)
    u2 = uhat - dt*k1 + 2.0*dt*k2
    k3 = f(u2)
    return uhat + dt*(k1 + 4.0*k2 + k3)/6.0

def energy(uhat, N):
    return 0.5*(uhat.abs()**2).sum()/(N**6)

def vorticity_hat(uhat, KX, KY, KZ):
    o0 = 1j*(KY*uhat[2] - KZ*uhat[1])
    o1 = 1j*(KZ*uhat[0] - KX*uhat[2])
    o2 = 1j*(KX*uhat[1] - KY*uhat[0])
    return torch.stack([o0, o1, o2], 0)

def to_real(uhat, dim=(1,2,3)):
    return torch.fft.ifftn(uhat, dim=dim).real

def tg_velocity(N, device, dtype=torch.float32):
    """Taylor-Green IC: u=(sin x cos y cos z, -cos x sin y cos z, 0)."""
    x = torch.linspace(0, 2*np.pi, N+1, device=device, dtype=dtype)[:-1]  # endpoint=False
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    ux = torch.sin(X)*torch.cos(Y)*torch.cos(Z)
    uy = -torch.cos(X)*torch.sin(Y)*torch.cos(Z)
    uz = torch.zeros_like(X)
    return torch.stack([ux, uy, uz])

def validate_tg(N=32, nu=0.01, T=5.0, dt=0.002, device='cuda'):
    """Validate against per-mode analytic decay (accounts for TG higher modes)."""
    dev = torch.device(device)
    KX, KY, KZ, K2, mask = make_k(N, dev)
    u0 = tg_velocity(N, dev).to(TORCH_DTYPE)
    uhat = project(torch.fft.fftn(u0, dim=(1,2,3)), KX, KY, KZ, K2, mask)
    E0 = energy(uhat, N)
    exact = 0.5*(uhat.abs()**2 * torch.exp(-2*nu*K2*T)).sum()/(N**6)
    nt = int(T/dt)
    t0 = time.time()
    for _ in range(nt):
        uhat = rk3_step(uhat, dt, nu, K2, N, TORCH_DTYPE, KX=KX, KY=KY, KZ=KZ, mask=mask)
    E_end = energy(uhat, N)
    rel = abs(E_end - exact)/E0
    print(f"[TG-GPU] N={N} nu={nu} T={T} dt={dt} steps={nt}", flush=True)
    print(f"  E0={E0.item():.8f} E_end={E_end.item():.8f} exact={exact.item():.8f}", flush=True)
    print(f"  rel_err={rel.item():.3e}  walltime={time.time()-t0:.1f}s", flush=True)
    return {'E0': E0.item(), 'E_end': E_end.item(), 'exact': exact.item(),
            'rel_err': rel.item(), 'walltime': time.time()-t0}

def random_divfree_ic(N, device, E_target=1.0, k0=2.5, sigma=1.5, seed=0, dtype=torch.float32):
    """Random divergence-free velocity IC with energy concentrated at LOW wavenumbers
    (large scales) via a Gaussian spectral envelope exp(-(k-k0)^2/sigma^2). This mimics
    an inertial-range-like entry and avoids the unphysical collapse seen with flat
    white-noise a_hat. Returns divergence-free-compatible uhat (via a_hat x k)."""
    dev = device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    KX, KY, KZ, K2, _ = make_k(N, dev)
    K = torch.sqrt(K2)
    # spectral envelope peaked near k0
    env = torch.exp(-(K - k0)**2 / (sigma**2)).to(torch.float32)
    env[0,0,0] = 0.0
    a = torch.randn((3,N,N,N), generator=g, device=dev, dtype=torch.float32) * env.unsqueeze(0)
    # u_hat = i k x a  (divergence-free)
    ux = 1j*(KY*a[2]-KZ*a[1])
    uy = 1j*(KZ*a[0]-KX*a[2])
    uz = 1j*(KX*a[1]-KY*a[0])
    uhat = torch.stack([ux,uy,uz],0).to(TORCH_DTYPE)
    # normalize to target energy
    cur = energy(uhat, N)
    uhat = uhat * (E_target/ (cur+1e-30))**0.5
    return uhat

def gen_trajectory(uhat0, N=64, nu=0.01, T=10.0, dt=0.002, n_snap=21, device='cuda',
                   force=None, seed=0):
    """Evolve an arbitrary divergence-free uhat0 and return vorticity snapshots."""
    dev = torch.device(device)
    KX, KY, KZ, K2, mask = make_k(N, dev)
    uhat = project(uhat0.clone(), KX, KY, KZ, K2, mask)
    nt = int(T/dt)
    step_per = max(1, nt//(n_snap-1))
    snaps=[]; times=[]
    for i in range(nt+1):
        if i % step_per == 0:
            om_r = to_real(vorticity_hat(uhat, KX, KY, KZ)).to(torch.float32)
            snaps.append(om_r.cpu().numpy()); times.append(round(i*dt,6))
        if i < nt:
            uhat = rk3_step(uhat, dt, nu, K2, N, TORCH_DTYPE, force=force, KX=KX,KY=KY,KZ=KZ,mask=mask)
    return np.stack(snaps), times

def gen_tg_trajectory(N=64, nu=0.01, T=10.0, dt=0.002, n_snap=21, device='cuda', seed=0):
    """Taylor-Green trajectory wrapper."""
    dev = torch.device(device)
    u0 = tg_velocity(N, dev)
    KX,KY,KZ,K2,mask = make_k(N, dev)
    uhat = project(torch.fft.fftn(u0.to(TORCH_DTYPE), dim=(1,2,3)), KX,KY,KZ,K2,mask)
    return gen_trajectory(uhat, N, nu, T, dt, n_snap, device)


if __name__ == '__main__':
    import sys
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(sys.argv) > 1 and sys.argv[1] == 'tg':
        N = int(sys.argv[2]) if len(sys.argv)>2 else 48
        r = validate_tg(N=N, device=dev)
        print(json.dumps(r))
    else:
        print("usage: python3 dns_solver_torch.py tg [N]")
