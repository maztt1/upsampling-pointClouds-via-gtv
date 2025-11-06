#!/usr/bin/env python3
"""
09__run_red_and_blue__.py — run 06→07 (RED & BLUE) → merge → multi-stage OUTLIER FILTER → save + view

New aggressive filter:
  1) ABS cap: remove points with max(|x|,|y|,|z|) > --abs-max (if given)
  2) Robust radius: distances from robust center (median) with MAD scaling; keep --radius-keep-pct
  3) kNN distance: keep --knn-keep-pct by k = --knn-k
  4) Iterate steps 2–3 for --iter-outlier-passes times

Defaults are conservative; you can turn it up via CLI.
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def run(cmd):
    print("[RUN]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def robust_center(points: np.ndarray):
    # component-wise median center
    return np.median(points, axis=0)

def mad(vals: np.ndarray):
    med = np.median(vals)
    return np.median(np.abs(vals - med)) + 1e-12

def stage_abs_cap(points: np.ndarray, labels: np.ndarray, abs_max: float | None):
    if abs_max is None:
        keep = np.ones(len(points), dtype=bool)
        return points, labels, keep, 0
    m = np.max(np.abs(points), axis=1) <= abs_max
    return points[m], labels[m], m, (~m).sum()

def stage_radius(points: np.ndarray, labels: np.ndarray, keep_pct: float, tau: float):
    # robust radius: ||(p - med)|| / MAD
    center = robust_center(points)
    dif = points - center
    # robust scale per-axis (MAD), then isotropic approx
    s = np.array([mad(dif[:,i]) for i in range(3)])
    s[s < 1e-12] = 1.0
    r = np.linalg.norm(dif / s, axis=1)
    thr = np.percentile(r, keep_pct) * float(tau)
    keep = r <= thr
    return points[keep], labels[keep], keep, (~keep).sum()

def stage_knn(points: np.ndarray, labels: np.ndarray, k: int, keep_pct: float):
    k = max(2, min(k, points.shape[0]-1))
    kd = cKDTree(points)
    d, _ = kd.query(points, k=k)
    # use the farthest neighbor among the k as the score
    score = d[:, -1] if d.ndim == 2 else d
    thr = np.percentile(score, keep_pct)
    keep = score <= thr
    return points[keep], labels[keep], keep, (~keep).sum()

def filter_outliers_multi(points: np.ndarray,
                          labels: np.ndarray,
                          abs_max: float | None,
                          radius_keep_pct: float,
                          radius_tau: float,
                          knn_k: int,
                          knn_keep_pct: float,
                          passes: int):
    N0 = points.shape[0]
    total_removed = 0
    keep_global = np.ones(N0, dtype=bool)

    # 1) ABS cap (one-shot)
    p, l, keep, removed = stage_abs_cap(points, labels, abs_max)
    # map back to original mask
    if abs_max is not None:
        idx = np.where(keep_global)[0]
        keep_global[idx[~keep]] = False
        total_removed += removed
        print(f"[FILTER] ABS cap removed {removed}")

    # 2–3) iterate robust radius + kNN
    for it in range(passes):
        p_oldN = p.shape[0]

        p, l, keep, rem = stage_radius(p, l, radius_keep_pct, radius_tau)
        idx = np.where(keep_global)[0]
        keep_global[idx[~keep]] = False
        total_removed += rem
        print(f"[FILTER] pass {it+1}: radius removed {rem}")

        if p.shape[0] < 10:
            print("[FILTER] too few points left after radius; stopping")
            break

        p, l, keep, rem = stage_knn(p, l, knn_k, knn_keep_pct)
        idx = np.where(keep_global)[0]
        keep_global[idx[~keep]] = False
        total_removed += rem
        print(f"[FILTER] pass {it+1}: kNN removed {rem}")

        if p.shape[0] >= p_oldN:
            # no progress; stop
            break

    kept = N0 - total_removed
    print(f"[FILTER] kept {kept}/{N0} points ({100.0*kept/max(1,N0):.3f}%)")
    return points[keep_global], labels[keep_global]

def main():
    ap = argparse.ArgumentParser("Run 06&07 (red+blue) → merge → aggressive filter → save + view")
    ap.add_argument("--preset", choices=["D","E"], required=True, help="ADMM param set")
    ap.add_argument("--low-res", required=True, help="Path to low-res .xyz (for Step 06)")
    ap.add_argument("--dir", default="intermediate", help="Intermediate folder")
    ap.add_argument("--skip-prepare", action="store_true", help="Skip Step 06")

    # Outlier filter controls
    ap.add_argument("--abs-max", type=float, default=None,
                    help="Drop points with max(|x|,|y|,|z|) > ABS_MAX (units of your data)")
    ap.add_argument("--radius-keep-pct", type=float, default=99.9,
                    help="Robust radius: keep this percentile (default 99.9)")
    ap.add_argument("--radius-tau", type=float, default=1.0,
                    help="Extra multiplier on the robust radius threshold (default 1.0)")
    ap.add_argument("--knn-k", type=int, default=8,
                    help="k for kNN distance (default 8)")
    ap.add_argument("--knn-keep-pct", type=float, default=99.9,
                    help="kNN: keep this percentile (default 99.9)")
    ap.add_argument("--iter-outlier-passes", type=int, default=2,
                    help="How many passes of (radius + kNN) to run (default 2)")
    args = ap.parse_args()

    PY   = sys.executable
    ROOT = Path(__file__).resolve().parent
    S06  = ROOT / "06__prepare_admm_matrices__.py"
    S07  = ROOT / "07__ADMM__solver__.py"
    if not S07.is_file(): sys.exit(f"[FATAL] missing {S07}")
    if (not args.skip_prepare) and (not S06.is_file()): sys.exit(f"[FATAL] missing {S06}")

    presets = {
        "D": {"lam":"2e-6", "rho":"25", "eps":"1e-2", "tol":"8e-4", "iters":"18"},
        "E": {"lam":"5e-6", "rho":"18", "eps":"5e-3", "tol":"6e-4", "iters":"18"},
    }
    p = presets[args.preset]

    # 06 + 07
    for colour in ("red","blue"):
        print("\n============================")
        print(f"[STEP] {colour.upper()}")
        print("============================")
        if not args.skip_prepare:
            run([PY, str(S06), "--opt", colour, "--dir", args.dir, "--low-res", args.low_res])
        run([PY, str(S07), "--opt", colour, "--dir", args.dir,
             "--lam", p["lam"], "--rho", p["rho"],
             "--eps-reg", p["eps"], "--tol", p["tol"], "--max-iters", p["iters"]])

    # Merge (safe)
    inter  = Path(args.dir)
    outdir = ROOT / "output"; outdir.mkdir(exist_ok=True)

    pts_all_path = inter / "points_all.xyz"
    pts = np.loadtxt(pts_all_path)          # (N,3)
    labels = np.load(inter / "labels.npy")  # (N,)
    N = pts.shape[0]
    if labels.shape[0] != N:
        sys.exit(f"[FATAL] labels size {labels.shape[0]} != points N {N}")

    def safe_apply(opt_path: Path, mask_val: int):
        if not opt_path.is_file(): return "absent"
        arr = np.loadtxt(opt_path)
        if arr.shape != pts.shape:
            print(f"[WARN] {opt_path.name} shape {arr.shape} != {pts.shape} — ignoring")
            return "ignored"
        mask = (labels == mask_val)
        pts[mask] = arr[mask]
        return "applied"

    red_status  = safe_apply(inter / "points_optim_red.xyz",  0)
    blue_status = safe_apply(inter / "points_optim_blue.xyz", 1)
    print(f"[INFO] applied red:{red_status}  blue:{blue_status}")

    # Aggressive multi-stage outlier filter
    pts_f, labels_f = filter_outliers_multi(
        pts, labels,
        abs_max=args.abs_max,
        radius_keep_pct=args.radius_keep_pct,
        radius_tau=args.radius_tau,
        knn_k=args.knn_k,
        knn_keep_pct=args.knn_keep_pct,
        passes=args.iter_outlier_passes
    )

    # Save XYZ (geometry only)
    xyz_path = outdir / "pointcloud_superres.xyz"
    np.savetxt(xyz_path, pts_f, fmt="%.6f")
    print(f"[INFO] saved XYZ → {xyz_path}  (N={pts_f.shape[0]})")

    # Save colored PLY + view
    ply_path = outdir / "pointcloud_superres_colored.ply"
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_f))
        colors = np.zeros((pts_f.shape[0], 3), dtype=float)
        colors[labels_f==0] = [1,0,0]
        colors[labels_f==1] = [0,0,1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=True)
        print(f"[INFO] saved colored PLY → {ply_path}")
        o3d.visualization.draw_geometries([pcd],
            window_name="Merged (aggressively filtered) — RED=red, BLUE=blue")
    except Exception as e:
        print(f"[INFO] viewer unavailable ({type(e).__name__}: {e}) — writing PLY manually")
        rgb = np.zeros((pts_f.shape[0],3), dtype=np.uint8)
        rgb[labels_f==0] = [255,0,0]; rgb[labels_f==1] = [0,0,255]
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts_f.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x,y,z),(r,g,b) in zip(pts_f, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        print(f"[INFO] saved colored PLY (no viewer) → {ply_path}")

    print("\n[ALL DONE] Optimised, merged, aggressively filtered, saved (XYZ+colored PLY), and shown if possible.")

if __name__ == "__main__":
    main()
