#!/usr/bin/env python3
"""
方案B — 3D 伪谱 DNS 求解器 (pseudo-spectral Navier-Stokes)
==========================================================
Standard pseudo-spectral incompressible Navier-Stokes solver on periodic box [0,2pi)^3.

Features:
  - 2/3 dealiasing (Orszag) + no truncation option
  - RK3 (low-storage) time integration
  - Spectral projection enforces div(u)=0 exactly at every step
  - Two flow regimes:
      (A) Taylor-Green decay (no forcing)          -> validates against analytic decay
      (B) Forced isotropic turbulence (spectral forcing back toward a target energy
          in a low-k band, simple deterministic 'forcing' : f_hat = eps*u_hat/|u|^2 in band)
  - Outputs velocity/vorticity snapshots on a downsampled grid for training data.

Vorticity computed in spectral domain: omega_hat = i k x u_hat.

Run modes:
  python3 dns_solver.py tg    --run Taylor-Green, print energy + compare analytic decay
  python3 dns_solver.py iso   --run forced isotropic, print spectra/statistics
  python3 dns_solver.py gen   --generate training corpus (velocity/vorticity pairs)
"""
import numpy as np
import json, os, time, hashlib

TWOPI = 2*np.pi

class PseudoSpectralDNS:
    def __init__(self, N=64, nu=0.01, dealias=True, force=None, dtype=np.complex128):
        self.N=N; self.nu=nu; self.dealias=dealias
        self.force=force  # dict for forced iso turbulence
        self.dtype=dtype
        # wavenumbers
        k = np.fft.fftfreq(N)*N
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = KX*KX+KY*KY+KZ*KZ
        self.K = np.sqrt(self.K2)
        self.KX, self.KY, self.KZ = KX, KY, KZ
        # dealias mask (2/3 rule)
        self.mask = (np.abs(KX) < 2*N/3) & (np.abs(KY) < 2*N/3) & (np.abs(KZ) < 2*N/3)
        # RHS scratch
        self.idx = slice(None)

    # ---- forcing helpers ----
    def _spectral_force(self, uhat, kmin, kmax, P_norm):
        """Deterministic forcing: maintain energy in shell [kmin,kmax] to target P_norm.
           f_hat = eps * u_hat / E_shell  (constant energy injection rate)."""
        K = self.K
        shell = (K>=kmin)&(K<kmax)
        Es = (np.abs(uhat)**2).sum(axis=0)[shell].sum()
        if Es < 1e-30: return np.zeros_like(uhat)
        eps = P_norm*Es  # injection proportional to shell energy (soft)
        f = np.zeros_like(uhat)
        for c in range(3):
            f[c][shell] = eps * uhat[c][shell] / (Es+1e-30)
        return f

    # ---- nonlinear term (dealised) ----
    def _nonlinear(self, uhat):
        N=self.N
        u = np.fft.ifftn(uhat, axes=(1,2,3))  # (3,N,N,N)
        if self.dealias:
            # pad/truncate -> use 3/2 zero-padding
            pad = N//2
            up = np.zeros((3, 3*N//2,)*3, dtype=self.dtype)
            up[:, :N, :N, :N] = uhat.copy()  # embiggen
            u3 = np.fft.ifftn(up, axes=(1,2,3))
            # derivative in padded space
            Kp = np.fft.fftfreq(3*N//2)*(3*N//2)
            KX,KY,KZ = np.meshgrid(Kp,Kp,Kp,indexing='ij')
            dudx = np.fft.fftn(np.fft.ifftn(1j*KX[None]*u3,axes=(1,2,3)),axes=(1,2,3)) if False else None
            # simpler: compute conv in physical then FFT back, dealias by truncation
            conv = np.zeros((3,N,N,N), dtype=self.dtype)
            u3f = np.fft.fftn(u3, axes=(1,2,3))
            # u.grad u : for each component
            grad_u = lambda v, ax: np.fft.ifftn(1j*KX[None]*v,axes=(1,2,3))
            return None  # placeholder, replaced below
        return u

    def nonlinear_rhs(self, uhat):
        """Compute -P( u.grad u ) via pseudospectral method with 3/2 dealiasing.
        Standard technique: zero-pad uhat from N to 3N/2, transform, multiply in
        physical space, transform back, truncate to N, then project div-free."""
        N=self.N; Mp=3*N//2
        # 1) zero-pad to 3N/2
        up = np.zeros((3,Mp,Mp,Mp), dtype=self.dtype)
        up[:, :N,:N,:N] = uhat
        kp = np.fft.fftfreq(Mp)*Mp
        KX,KY,KZ = np.meshgrid(kp,kp,kp,indexing='ij')
        # 2) physical velocity on padded grid
        w = np.fft.ifftn(up, axes=(1,2,3))       # (3,Mp,Mp,Mp) complex
        # 3) physical gradients (spectral on padded grid)
        gw = [np.fft.ifftn(1j*KX[None]*up, axes=(1,2,3)),
              np.fft.ifftn(1j*KY[None]*up, axes=(1,2,3)),
              np.fft.ifftn(1j*KZ[None]*up, axes=(1,2,3))]
        # 4) (u.grad u)_i = u_x du_i/dx + u_y du_i/dy + u_z du_i/dz
        nlin = w[0:1]*gw[0] + w[1:2]*gw[1] + w[2:3]*gw[2]
        # 5) FFT back and truncate to N (dealias), then project
        adv_hat = np.fft.fftn(nlin, axes=(1,2,3))[:, :N,:N,:N]
        return self.project(adv_hat)

    def project(self, uhat):
        """Project divergence-free: u_hat <- (I - k k^T/k^2) u_hat (k=0 set 0)."""
        N=self.N
        KX,KY,KZ=self.KX,self.KY,self.KZ
        K2=self.K2.copy(); K2[0,0,0]=1.0
        kdotu = KX*uhat[0]+KY*uhat[1]+KZ*uhat[2]
        uhat = uhat - np.stack([KX*kdotu, KY*kdotu, KZ*kdotu],axis=0)/K2
        if self.dealias:
            uhat = uhat * self.mask[None]
        uhat[:,0,0,0]=0.0
        return uhat

    def rhs(self, uhat):
        """du_hat/dt = -P(u.grad u) - nu k^2 u_hat  (+ forcing)"""
        nl = self.nonlinear_rhs(uhat)
        rhs = nl - self.nu*self.K2*uhat
        if self.force is not None:
            f = self._spectral_force(uhat, self.force['kmin'], self.force['kmax'], self.force['P'])
            rhs = rhs + f
        return rhs

    def step(self, uhat, dt):
        """Classic 3rd-order Runge-Kutta (RK3).
           k1 = f(u); k2 = f(u+dt/2 k1); k3 = f(u - dt k1 + 2 dt k2)
           u' = u + dt/6 (k1 + 4 k2 + k3)"""
        k1 = self.rhs(uhat)
        u1 = uhat + dt*0.5*k1
        k2 = self.rhs(u1)
        u2 = uhat - dt*k1 + 2.0*dt*k2
        k3 = self.rhs(u2)
        return uhat + dt*(k1 + 4.0*k2 + k3)/6.0

    # ---- helpers for real fields ----
    def to_real(self, uhat):
        return np.fft.ifftn(uhat, axes=(1,2,3)).real

    def vorticity_hat(self, uhat):
        KX,KY,KZ=self.KX,self.KY,self.KZ
        o0 = 1j*(KY*uhat[2]-KZ*uhat[1])
        o1 = 1j*(KZ*uhat[0]-KX*uhat[2])
        o2 = 1j*(KX*uhat[1]-KY*uhat[0])
        return np.stack([o0,o1,o2],axis=0)

    def energy(self, uhat):
        N=self.N
        return 0.5*(np.abs(uhat)**2).sum()/(N**6)

    def enstrophy(self, uhat):
        N=self.N
        oh=self.vorticity_hat(uhat)
        return 0.5*(np.abs(oh)**2).sum()/(N**6)

    def spectrum(self, uhat):
        N=self.N
        k=np.fft.fftfreq(N)*N
        KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
        K=np.round(np.sqrt(KX**2+KY**2+KZ**2)).astype(int)
        spec=np.zeros(N//2+1)
        for c in range(3):
            A=np.abs(uhat[c])**2
            for kk in range(1,N//2):
                spec[kk]+=A[K==kk].sum()
        # E(k) = 1/2 * sum shell, normalized by N^6
        return spec*0.5/(N**6)  # 3D shell-summed

# ============================================================
#  Taylor-Green exact decay
def tg_pressure_free_ic(N):
    """Taylor-Green initial velocity (divergence-free):
       u = (sin x cos y cos z, -cos x sin y cos z, 0)."""
    x=np.linspace(0,TWOPI,N,endpoint=False)
    X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
    ux=np.sin(X)*np.cos(Y)*np.cos(Z)
    uy=-np.cos(X)*np.sin(Y)*np.cos(Z)
    uz=np.zeros_like(X)
    u=np.stack([ux,uy,uz])
    return u

def run_tg(N=64, nu=0.01, T=10.0, dt=0.002):
    print(f"[TG] N={N} nu={nu} T={T} dt={dt}", flush=True)
    dns=PseudoSpectralDNS(N=N, nu=nu, dealias=True)
    u0=tg_pressure_free_ic(N)
    uhat=np.fft.fftn(u0, axes=(1,2,3))
    uhat=dns.project(uhat)
    E0=dns.energy(uhat)
    print(f"  initial energy={E0:.6f}, 1/8={0.125:.6f} (expect ~0.125? actually TG energy = 1/4? check)", flush=True)
    E_analytic = E0*np.exp(-2*nu*T)  # linear decay of each Fourier mode
    nt=int(T/dt)
    # record
    En=[dns.energy(uhat)]
    for i in range(1,nt+1):
        uhat=dns.step(uhat, dt)
        if i%int(nt/20)==0 or i==nt:
            En.append(dns.energy(uhat))
    E_end=En[-1]
    # analytic decay energy factor e^{-2nu T}
    factor=np.exp(-2*nu*T)
    err=abs(E_end-E0*factor)/E0
    print(f"  final E={E_end:.6f}, analytic E0*e^-2nuT={E0*factor:.6f}, rel_err={err:.2e}", flush=True)
    return {'E0':E0,'E_end':E_end,'expected':E0*factor,'rel_err':err,
            'energy_curve':[float(e) for e in En]}

if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'tg'
    if mode=='tg':
        r=run_tg()
        print("TG validation:", json.dumps({k:v for k,v in r.items() if k!='energy_curve'}, indent=1), flush=True)
    elif mode=='iso':
        print("ISO mode not yet wired in __main__", flush=True)
    else:
        print("unknown", mode)
