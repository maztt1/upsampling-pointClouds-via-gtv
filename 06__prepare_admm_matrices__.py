#!/usr/bin/env python3
"""
Step 05 — Partition graph into RED / BLUE sets

Goal
----
Assign 0/1 labels so every vertex has at least `min_cross` neighbors with the
opposite color (default 2). The original edge set is preserved.

Why v2.4?
---------
Earlier versions deleted same-color edges to force bipartiteness, which broke
later steps that expect intra-set edges. Dinesh et al. only require the
cross-neighbor guarantee, not a strictly bipartite graph. v2.4 keeps all edges
and flips colors until the cross-degree constraint is satisfied.

Outputs
-------
- labels.npy          (N,) int8  0=red, 1=blue
- isolated_nodes.npy  (K,) int    nodes with total degree < `min_cross`
- edges_final.npy     (M,2) int   copy of original edge list
- weights_final.npy   (M,) float  copy of original weights

Usage
-----
    python 05__bipartite_partitioning_v2.py [--min-cross 2] [--view]

Algorithm (per iteration O(|E|))
-------------------------------
1) Initial 2-coloring via BFS (no edge removal).
2) Recompute cross-degree; flip nodes whose flipped cross-degree meets `min_cross`.
3) Stop when no flips occur. Nodes with degree < `min_cross` are reported as isolated.
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np
import open3d as o3d
from collections import deque
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_graph(dir_path: Path):
    pts  = np.loadtxt(dir_path / "points_all.xyz")
    E    = np.load(dir_path / "edges.npy").astype(int)
    W    = np.load(dir_path / "weights.npy")
    return pts, E, W


def save_outputs(dir_path: Path, labels: np.ndarray, E: np.ndarray, W: np.ndarray, iso: np.ndarray):
    np.save(dir_path / "labels.npy", labels.astype(np.int8))
    np.save(dir_path / "edges_final.npy", E)
    np.save(dir_path / "weights_final.npy", W)
    np.save(dir_path / "isolated_nodes.npy", iso)
    print(f"[INFO] saved labels / edges_final / weights_final in {dir_path}")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def partition(pts: np.ndarray, E: np.ndarray, min_cross: int = 2):
    N = len(pts)

    # Build adjacency
    adj: List[List[int]] = [[] for _ in range(N)]
    for u, v in E:
        adj[u].append(v)
        adj[v].append(u)

    # Initial BFS coloring
    labels = -np.ones(N, dtype=np.int8)
    for root in range(N):
        if labels[root] != -1:
            continue
        labels[root] = 0  # red
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            cu = labels[u]
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = 1 - cu  # opposite color
                    q.append(v)
    print("[INFO] initial colouring done (BFS)")

    # Enforce cross-degree via flips
    changed = True
    passes = 0
    while changed:
        changed = False
        passes += 1
        cross_deg = np.zeros(N, dtype=int)
        deg       = np.fromiter((len(adj[i]) for i in range(N)), dtype=int, count=N)
        for u, v in E:
            if labels[u] != labels[v]:
                cross_deg[u] += 1
                cross_deg[v] += 1

        # Nodes that would meet threshold after flipping
        to_flip = []
        for i in range(N):
            if cross_deg[i] >= min_cross:
                continue
            if deg[i] - cross_deg[i] >= min_cross:
                to_flip.append(i)
        if to_flip:
            labels[to_flip] ^= 1
            changed = True
            print(f"[DEBUG] pass {passes}: flipped {len(to_flip)} vertices")

    # Nodes impossible to fix due to low total degree
    iso = np.flatnonzero(deg < min_cross)
    if len(iso):
        print(f"[WARN] {len(iso)} vertices totally isolated (deg < {min_cross}) – written to isolated_nodes.npy")
    print(f"[INFO] partition finished in {passes} passes")
    return labels, iso


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Step 05 — partition RED/BLUE sets (v2.4)")
    ap.add_argument("--dir", default="intermediate", help="directory containing points_all.xyz, edges.npy, weights.npy")
    ap.add_argument("--min-cross", type=int, default=2, help="required opposite-colour neighbours (default 2)")
    ap.add_argument("--view", action="store_true", help="visualise  colouring in Open3D viewer")
    args = ap.parse_args()

    d = Path(args.dir)
    required = [d / "points_all.xyz", d / "edges.npy", d / "weights.npy"]
    for f in required:
        if not f.is_file():
            sys.exit(f"[ERROR] missing required file: {f}")

    pts, E, W = load_graph(d)
    labels, iso = partition(pts, E, min_cross=args.min_cross)
    save_outputs(d, labels, E, W, iso)

    # Optional viewer
    if args.view:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        colours = np.zeros((len(pts), 3))
        colours[labels == 0] = [1, 0, 0]
        colours[labels == 1] = [0, 0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colours)
        o3d.visualization.draw_geometries([pcd], window_name="RED / BLUE partition (v2.4)")

if __name__ == "__main__":
    main()
