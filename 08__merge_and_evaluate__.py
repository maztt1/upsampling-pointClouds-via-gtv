#!/usr/bin/env python3
"""
08__merge_and_evaluate__.py
===========================
After running ADMM optimization for one or both sets (RED / BLUE), this script:

1. Builds the final full point cloud (optimized + fixed points).
2. Compares it with the original low-resolution input and,
   if available, a high-resolution ground-truth reference.
   Computes Chamfer and Hausdorff distances.
3. Saves a final *.xyz* (or *.ply*) point cloud for visualization
   in CloudCompare/Open3D.
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
import os, sys
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DIR_INTER   = "intermediate"   # folder with intermediate results
DIR_OUTPUT  = "output"         # folder for final outputs

LOW_RES_XYZ = "input/Asterix_downsampled_30pct.xyz"    # low-resolution point cloud
HI_RES_XYZ  = "input/Asterix_original.xyz"             # high-resolution reference (optional)

os.makedirs(DIR_OUTPUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
pts_all   = np.loadtxt(os.path.join(DIR_INTER, "points_all.xyz"))   # (N,3)
labels    = np.load(os.path.join(DIR_INTER, "labels.npy"))          # (N,)

# If optimized sets exist, overwrite corresponding coordinates
try:
    pts_red_opt = np.loadtxt(os.path.join(DIR_INTER, "points_optim_red.xyz"))
    pts_all[labels == 0] = pts_red_opt[labels == 0]
except FileNotFoundError:
    print("[WARN] points_optim_red.xyz not found – RED set assumed unchanged")

try:
    pts_blue_opt = np.loadtxt(os.path.join(DIR_INTER, "points_optim_blue.xyz"))
    pts_all[labels == 1] = pts_blue_opt[labels == 1]
except FileNotFoundError:
    pass  # BLUE may not have been optimized

# ---------------------------------------------------------------------------
# Save final point cloud
# ---------------------------------------------------------------------------
final_xyz = os.path.join(DIR_OUTPUT, "pointcloud_superres.xyz")
np.savetxt(final_xyz, pts_all, fmt="%.6f")
print(f"[INFO] final point cloud → {final_xyz}  (N={len(pts_all)})")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
print("[INFO] computing distance metrics …")
low_res = np.loadtxt(LOW_RES_XYZ)
hi_res  = None
if os.path.isfile(HI_RES_XYZ):
    hi_res = np.loadtxt(HI_RES_XYZ)

# Chamfer distance helper
def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    kd = cKDTree(b)
    dist, _ = kd.query(a, k=1)
    return float(dist.mean())

cd_lr  = chamfer(pts_all, low_res)
print(f"Chamfer( superres ↔ low-res )  = {cd_lr:.6f}")

if hi_res is not None:
    cd_hr  = chamfer(pts_all, hi_res)
    hd_hr  = max(chamfer(hi_res, pts_all), chamfer(pts_all, hi_res))
    print(f"Chamfer( superres ↔ high-res ) = {cd_hr:.6f}")
    print(f"Hausdorff( superres, high-res ) = {hd_hr:.6f}")
else:
    print("[WARN] high-res reference not found – skipping GT metrics")

# ---------------------------------------------------------------------------
# Optional visualization
# ---------------------------------------------------------------------------
try:
    pcd_final = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_all))
    pcd_final.paint_uniform_color([1, 0, 0])   # red = super-res
    pcd_low   = o3d.io.read_point_cloud(LOW_RES_XYZ, format='xyz')
    pcd_low.paint_uniform_color([0, 0, 1])     # blue = low-res
    o3d.visualization.draw_geometries(
        [pcd_low, pcd_final],
        window_name="Blue = Low-Res  |  Red = Super-Res"
    )
except Exception as e:
    print("[WARN] Open3D visualisation skipped:", e)
