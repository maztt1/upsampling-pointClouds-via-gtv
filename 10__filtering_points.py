#!/usr/bin/env python3
"""
10__filter_by_lowres_proximity.py — keep only super-res points near the low-res cloud.

Radius R is computed from the low-res characteristic spacing:
    d_low = distance of each low-res point to its nearest *other* low-res point
    R = --k * median(d_low)            (default k=6.0)

We then keep super-res points whose nearest distance to the low-res cloud <= R.

This NEVER moves points; it only removes the far-away "infinity" fliers.

Usage:
  python 10__filter_by_lowres_proximity.py --super output/pointcloud_superres.xyz \
      --low input/Asterix_downsampled_30pct.xyz --k 6 --labels intermediate/labels.npy --view
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x,y,z),(r,g,b) in zip(pts, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def main():
    ap = argparse.ArgumentParser("Filter super-res points by proximity to low-res")
    ap.add_argument("--super", required=True, help="Super-res XYZ (e.g., output/pointcloud_superres.xyz)")
    ap.add_argument("--low", required=True, help="Low-res XYZ (original)")
    ap.add_argument("--k", type=float, default=6.0, help="Radius multiplier (R = k * median low-res spacing)")
    ap.add_argument("--out", default="output/pointcloud_superres_nearLOW.xyz", help="Output XYZ path")
    ap.add_argument("--ply", default="output/pointcloud_superres_nearLOW_colored.ply", help="Output colored PLY")
    ap.add_argument("--labels", default=None, help="Optional labels.npy for coloring (RED=red, BLUE=blue)")
    ap.add_argument("--view", action="store_true", help="Show viewer (requires open3d)")
    args = ap.parse_args()

    super_path = Path(args.super); low_path = Path(args.low)
    if not super_path.is_file(): sys.exit(f"[ERROR] missing super-res: {super_path}")
    if not low_path.is_file():   sys.exit(f"[ERROR] missing low-res:   {low_path}")

    S = np.loadtxt(super_path)  # (Ns,3)
    L = np.loadtxt(low_path)    # (Nl,3)
    if S.ndim!=2 or S.shape[1]!=3 or L.ndim!=2 or L.shape[1]!=3:
        sys.exit("[ERROR] expecting XYZ format (N,3)")

    # 1) characteristic spacing of low-res
    kdL = cKDTree(L)
    dL, _ = kdL.query(L, k=min(2, max(1, len(L)-1)))
    # dL[:,0]=0 (self); take the second column if available
    spacing = dL[:,1] if dL.ndim==2 and dL.shape[1]>1 else dL[:,0]
    med_spacing = float(np.median(spacing) + 1e-12)
    R = float(args.k) * med_spacing
    print(f"[INFO] median low-res spacing = {med_spacing:.6g} → R = k*median = {args.k}× = {R:.6g}")

    # 2) distance of super-res points to low-res
    dS, _ = kdL.query(S, k=1)
    keep = dS <= R
    kept = int(keep.sum()); removed = int(len(S)-kept)
    print(f"[INFO] kept {kept}/{len(S)} points (removed {removed})")

    S_keep = S[keep]

    # 3) save XYZ
    out_xyz = Path(args.out)
    np.savetxt(out_xyz, S_keep, fmt="%.6f")
    print(f"[INFO] saved XYZ → {out_xyz}")

    # 4) colored PLY (labels optional)
    rgb = np.full((kept,3), 200, dtype=np.uint8)
    if args.labels:
        try:
            labels = np.load(args.labels)
            if labels.shape[0] == S.shape[0]:
                lab_keep = labels[keep]
                rgb[lab_keep==0] = [255,0,0]  # RED set
                rgb[lab_keep==1] = [0,0,255]  # BLUE set
        except Exception as e:
            print("[WARN] labels load failed; using gray", e)

    out_ply = Path(args.ply)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S_keep))
        pcd.colors = o3d.utility.Vector3dVector((rgb/255.0).astype(np.float32))
        o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=True)
        print(f"[INFO] saved colored PLY → {out_ply}")
        if args.view:
            o3d.visualization.draw_geometries([pcd], window_name="Near-low-res filtered")
    except Exception:
        write_ply(out_ply, S_keep, rgb)
        print(f"[INFO] saved colored PLY (manual) → {out_ply}")

if __name__ == "__main__":
    main()
if yes but I can do it I think I wrote the module for that