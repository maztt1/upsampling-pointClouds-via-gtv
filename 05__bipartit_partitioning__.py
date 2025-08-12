#!/usr/bin/env python3
"""
Step 05 – Greedy bipartite graph approximation (v2)
=================================================
Improved implementation that respects edge weights and guarantees
≥ 2 neighbours of the opposite colour **for _both_ partitions**.
The algorithm still follows a greedy spirit (cf. Zeng et al., 2017)
but makes "smarter" local decisions:

* **Conflict resolution**: when an edge connects two nodes of the same
  colour, we either _remove the lighter edge_ or _flip_ the colour of
  the vertex with the lower weighted degree – whichever keeps more
  total edge weight.
* **Stability**: each vertex may flip its colour **at most once** to
  avoid endless flip‑flop oscillations.
* **Neighbourhood constraint**: after the initial BFS pass we iterate
  until every node has at least two neighbours of the opposite colour.
* **Outputs**
  - ``labels.npy``
  - ``edges_final.npy``dges that remain in the bipartite graph
  - ``weights_final.npy`` weight per edge (aligned with ``edges_final``)
  - ``removed_edges.npy`` edges pruned during the procedure (for statistics)

Run ``python 05__bipartite_partitioning_v2.py --help`` for options.
"""
from __future__ import annotations

import argparse, os, sys
from collections import deque
from typing import Dict, List, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _build_adj(n: int, edges: np.ndarray, weights: np.ndarray):
    """Return adjacency as list[dict[int,float]] for fast weight access."""
    adj: List[Dict[int, float]] = [dict() for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        if u == v:
            continue
        adj[u][v] = w
        adj[v][u] = w
    return adj


def _remove_edge(adj, u: int, v: int):
    """Remove undirected edge (u,v) from adjacency in‑place."""
    if v in adj[u]:
        del adj[u][v]
    if u in adj[v]:
        del adj[v][u]

# -----------------------------------------------------------------------------
# Bipartite partitioning (weighted greedy)
# -----------------------------------------------------------------------------

def bipartite_partition(
    n_points: int,
    edges: np.ndarray,
    weights: np.ndarray,
    ensure_two_each: bool = True,
    verbose: bool = True,
):
    """Return labels (0/1), edges_final, weights_final, removed_edges."""

    # ⮞ adjacency -------------------------------------------------------------
    adj = _build_adj(n_points, edges, weights)

    labels = -np.ones(n_points, dtype=int)          # -1 = uncoloured
    flipped = np.zeros(n_points, dtype=bool)        # lock after 1 flip
    removed: List[Tuple[int, int]] = []

    median_w = np.median(weights) if len(weights) else 0.0

    # pass 1 – BFS colouring with weighted conflict resolution ---------------
    for root in range(n_points):
        if labels[root] != -1:
            continue
        labels[root] = 0
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            cu = labels[u]
            for v in list(adj[u].keys()):
                w_uv = adj[u][v]
                if labels[v] == -1:
                    labels[v] = 1 - cu
                    q.append(v)
                elif labels[v] == cu:  # conflict
                    # strategy: keep more weight -> either flip vertex or drop edge
                    du = sum(adj[u].values())
                    dv = sum(adj[v].values())
                    # cost of flipping vs removing
                    flip_cost_u = sum(adj[u][nbr] for nbr in adj[u] if labels[nbr] == cu) if not flipped[u] else np.inf
                    flip_cost_v = sum(adj[v][nbr] for nbr in adj[v] if labels[nbr] == cu) if not flipped[v] else np.inf
                    remove_cost = w_uv
                    # choose minimal cost action
                    action = np.argmin([flip_cost_u, flip_cost_v, remove_cost])
                    if action == 0:  # flip u
                        labels[u] = 1 - cu
                        flipped[u] = True
                        # restarting BFS from u keeps queue manageable
                        q.append(u)
                    elif action == 1:  # flip v
                        labels[v] = 1 - cu
                        flipped[v] = True
                        q.append(v)
                    else:  # remove edge
                        _remove_edge(adj, u, v)
                        removed.append((u, v))

    if verbose:
        print(f"[INFO] pass 1: coloured graph with {len(removed)} edges removed due to conflicts")

    # pass 2 – guarantee ≥2 opposite neighbours for **both** colours ----------
    if ensure_two_each:
        changed = True
        it = 0
        while changed and it < 10:  # safety cap
            changed = False
            it += 1
            for i in range(n_points):
                opp_nbrs = [j for j in adj[i] if labels[j] != labels[i]]
                if len(opp_nbrs) >= 2:
                    continue  # ok
                # attempt flip if not already flipped
                if not flipped[i]:
                    labels[i] = 1 - labels[i]
                    flipped[i] = True
                    changed = True
                    # after flip, remove conflicts
                    for j in list(adj[i].keys()):
                        if labels[j] == labels[i]:
                            _remove_edge(adj, i, j)
                            removed.append((i, j))
                    continue
                # cannot flip → drop lightest same‑colour edges until ≥2 opp‑nbrs
                same_nbrs = sorted(
                    [(j, w) for j, w in adj[i].items() if labels[j] == labels[i]],
                    key=lambda t: t[1]
                )
                while len(opp_nbrs) < 2 and same_nbrs:
                    j, _ = same_nbrs.pop(0)
                    _remove_edge(adj, i, j)
                    removed.append((i, j))
                    opp_nbrs = [k for k in adj[i] if labels[k] != labels[i]]
                    changed = True
            if verbose and changed:
                print(f"[INFO] pass 2‑iter{it}: adjusting graph for ≥2‑neighbour rule")

    # ---------------------------------------------------------------------
    # collect final edges & weights
    edges_final, weights_final = [], []
    for u in range(n_points):
        for v, w in adj[u].items():
            if u < v:  # store each undirected edge once
                edges_final.append((u, v))
                weights_final.append(w)

    return (
        labels,
        np.array(edges_final, dtype=int),
        np.array(weights_final, dtype=float),
        np.array(removed, dtype=int),
    )

# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Step 05 – bipartite partitioning (v2, weighted)")
    parser.add_argument("--points", default="intermediate/points_all.xyz", help="XYZ file with all points (N×3)")
    parser.add_argument("--edges", default="intermediate/edges.npy", help="npy file with undirected edges (M×2)")
    parser.add_argument("--weights", default="intermediate/weights.npy", help="npy file with edge weights (M,)")
    parser.add_argument("--outdir", default="intermediate", help="directory to store outputs")
    parser.add_argument("--view", action="store_true", help="visualise red/blue sets in Open3D viewer")
    args = parser.parse_args()

    # -- load data ------------------------------------------------------------
    if not os.path.isfile(args.points):
        sys.exit(f"[ERROR] points file not found: {args.points}")
    if not os.path.isfile(args.edges):
        sys.exit(f"[ERROR] edges file not found: {args.edges}")
    if not os.path.isfile(args.weights):
        sys.exit(f"[ERROR] weights file not found: {args.weights}")

    pts = np.loadtxt(args.points)
    E = np.load(args.edges).astype(int)
    W = np.load(args.weights).astype(float)

    labels, E_final, W_final, removed = bipartite_partition(len(pts), E, W, True, True)

    # -- save -----------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "labels.npy"), labels)
    np.save(os.path.join(args.outdir, "edges_final.npy"), E_final)
    np.save(os.path.join(args.outdir, "weights_final.npy"), W_final)
    np.save(os.path.join(args.outdir, "removed_edges.npy"), removed)

    print(f"[INFO] partition done → red {np.sum(labels==0)}, blue {np.sum(labels==1)}")
    print(f"[INFO] final graph  → {len(E_final)} edges (removed {len(removed)})")
    print(f"[INFO] outputs saved in {args.outdir}")

    # optional visualisation --------------------------------------------------
    if args.view:
        try:
            import open3d as o3d
        except ImportError:
            sys.exit("[ERROR] open3d not installed; pip install open3d for --view")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        colours = np.zeros((len(pts), 3))
        colours[labels == 0] = [1, 0, 0]  # red
        colours[labels == 1] = [0, 0, 1]  # blue
        pcd.colors = o3d.utility.Vector3dVector(colours)
        o3d.visualization.draw_geometries([pcd], window_name="Bipartite sets: Red / Blue (v2)")

if __name__ == "__main__":
    main()
