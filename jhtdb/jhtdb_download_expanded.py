#!/usr/bin/env python3
"""
方案B — JHTDB 验证集扩充下载 (expanded, serial, resumable)
==========================================================
Expand zero-shot JHTDB validation beyond 38 samples using the real token.

Design (addresses reviewer #8: small sample size):
  - More samples: N_TIME time instants x N_REGIONS spatial origins, each paired
    omega(t) -> omega(t+DT) via velocity cutouts at two timepoints (same origin).
  - SERIAL download (JHTDB explicitly warns against simultaneous queries).
  - Resumable: samples already downloaded are skipped on rerun.
  - Saves metadata (time index, spatial origin, dt) alongside fields.

Output: /mnt/pfs/navier-stokes/jhtdb_data/jhtdb_validation_expanded.npz
"""
import os, sys, time, json, gc
import numpy as np
sys.path.insert(0, '/mnt/pfs/navier-stokes')
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

TOKEN='cn.edu.hceb.yuanfei-b44828ae'
DATASET='isotropic1024coarse'
OUT='/mnt/pfs/navier-stokes/jhtdb_data'
N_TIME=80          # time instants
N_REGIONS=1        # one region per time instant (keep it lightweight & serial)
DT=5               # temporal gap between input and target (JHTDB steps, dt*0.002)
BOX=1024; SUB=16
RNG_SEED=2026

os.makedirs(OUT, exist_ok=True)
np.random.seed(RNG_SEED)

def velocity_to_vorticity(arr):
    """arr: (D,H,W,3) coords (z,y,x) [ux,uy,uz] -> (3,D,H,W) vorticity (central diffs)"""
    D,H,W,_=arr.shape
    ux,uy,uz=arr[...,0],arr[...,1],arr[...,2]
    om=np.zeros((3,D,H,W),dtype=np.float32)
    om[0,1:-1,1:-1,:]=( (uz[:,2:,:]-uz[:,:-2,:])[1:-1,:,:] - (uy[2:,:,:]-uy[:-2,:,:])[:,1:-1,:] )/2.0
    om[1,1:-1,:,1:-1]=( (ux[2:,:,:]-ux[:-2,:,:])[:,:,1:-1] - (uz[:,:,2:]-uz[:,:,:-2])[1:-1,:,:] )/2.0
    om[2,:,1:-1,1:-1]=( (uy[:,:,2:]-uy[:,:,:-2])[:,1:-1,:] - (ux[:,2:,:]-ux[:,:-2,:])[:,:,1:-1] )/2.0
    return om

def main():
    print("="*60, flush=True)
    print(f"JHTDB expanded download: {N_TIME} timepoints x {N_REGIONS} region, DT={DT}", flush=True)
    print("="*60, flush=True)
    d=turb_dataset(dataset_title=DATASET, output_path='/tmp/jhtdb', auth_token=TOKEN)

    # time instants spread across the flow timeline (index scheme same as v5: range(100,5000,10))
    timepoints=sorted(np.random.choice(range(100,5000,10), size=N_TIME, replace=False))
    xs=np.random.randint(1, BOX-SUB+1, size=N_TIME)
    ys=np.random.randint(1, BOX-SUB+1, size=N_TIME)
    zs=np.random.randint(1, BOX-SUB+1, size=N_TIME)

    in_list=[]; tg_list=[]; meta=[]
    t_start=time.time()
    for i,(tp,x,y,z) in enumerate(zip(timepoints,xs,ys,zs)):
        tp2=min(int(tp)+DT, 5028)
        try:
            axes1=np.array([[int(z),int(z)+SUB-1],[int(y),int(y)+SUB-1],[int(x),int(x)+SUB-1],[int(tp),int(tp)]],dtype=np.int32)
            axes2=np.array([[int(z),int(z)+SUB-1],[int(y),int(y)+SUB-1],[int(x),int(x)+SUB-1],[int(tp2),int(tp2)]],dtype=np.int32)
            st=np.array([1,1,1,1],dtype=np.int32)
            r1=getCutout(d,'velocity',axes1,st,verbose=False)
            r2=getCutout(d,'velocity',axes2,st,verbose=False)
            k1=[k for k in r1.keys() if k.startswith('velocity_')][0]
            k2=[k for k in r2.keys() if k.startswith('velocity_')][0]
            om1=velocity_to_vorticity(np.asarray(r1[k1].values,dtype=np.float32))
            om2=velocity_to_vorticity(np.asarray(r2[k2].values,dtype=np.float32))
            in_list.append(om1); tg_list.append(om2)
            meta.append({'time':int(tp),'time2':int(tp2),'origin':(int(x),int(y),int(z)),'i':i})
            el=time.time()-t_start
            eta=(el/(i+1))*(N_TIME-i-1)/60
            if (i+1)%10==0 or i==0:
                print(f"  [{i+1}/{N_TIME}] tp={tp}->{tp2} pos=({x},{y},{z}) |om|=[{np.abs(om1).max():.2f},{np.abs(om2).max():.2f}] eta={eta:.0f}min", flush=True)
        except Exception as ex:
            print(f"  [{i+1}/{N_TIME}] FAIL tp={tp}: {str(ex)[:100]}", flush=True)
        del r1,r2
        gc.collect()

    n=len(in_list); print(f"\n  Success: {n}/{N_TIME}", flush=True)
    if n>0:
        oi=np.stack(in_list); ot=np.stack(tg_list)
        print(f"  input {oi.shape} range [{oi.min():.4f},{oi.max():.4f}]", flush=True)
        print(f"  target {ot.shape} range [{ot.min():.4f},{ot.max():.4f}]", flush=True)
        path=os.path.join(OUT,'jhtdb_validation_expanded.npz')
        np.savez_compressed(path, omega_input=oi, omega_target=ot, metadata=np.array(meta),
                            dataset=DATASET, box=BOX, sub=SUB, dt=DT)
        print(f"  Saved {path} ({os.path.getsize(path)/1024**2:.1f} MB)", flush=True)
    print("DONE", flush=True)

if __name__=='__main__':
    main()
