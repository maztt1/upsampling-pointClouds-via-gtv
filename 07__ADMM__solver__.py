#!/usr/bin/env python3
"""
Step 07 — ADMM solver for Graph-TV on surface normals
=============================================================
self-contained implementation of the optimization in Eq. (5)
from Dinesh et al., 2019.

What's new in v2.2
------------------
* Shape-safe edge weights — the length of `w_edge` is guaranteed to match
  the number of 3-row edge blocks in matrix **B** (i.e., rows_B/3) to avoid
  broadcasting errors. Logic:
  1) If `weights_rr_<set>.npy` exists: load it; if mismatched, trim or raise.
  2) Otherwise, derive from `edges_final.npy`/`weights_final.npy` by selecting
     only edges whose endpoints both have the target color and are not listed
     in `isolated_nodes.npy` (if present); identical to Step-06 filtering.
     Save the result for reuse.
* Clearer INFO messages about weight lengths and consistency with **B**.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_sparse(path: str) -> sp.csr_matrix:
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] file not found: {path}")
    return sp.load_npz(path).tocsr()

def build_weights_rr(dir_path: str, opt: str, labels: np.ndarray) -> np.ndarray:
    """Build weights for intra-set edges (RR or BB) using the same filtering as Step-06:
    keep only edges fully inside the chosen color set and exclude isolated nodes."""
    edges   = np.load(os.path.join(dir_path, "edges_final.npy"))   # (E,2)
    weights = np.load(os.path.join(dir_path, "weights_final.npy")) # (E,)

    # Isolated nodes may be absent
    iso_path = os.path.join(dir_path, "isolated_nodes.npy")
    isolated = np.load(iso_path) if os.path.isfile(iso_path) else np.array([], dtype=int)
    mask_iso = np.zeros(labels.shape[0], dtype=bool)
    mask_iso[isolated] = True

    set_id   = 0 if opt == "red" else 1
    mask_set = labels == set_id

    mask_rr = (
        mask_set[edges[:, 0]] & mask_set[edges[:, 1]] &
        ~mask_iso[edges[:, 0]] & ~mask_iso[edges[:, 1]]
    )
    return weights[mask_rr]

def ensure_weights_rr(dir_path: str, opt: str, labels: np.ndarray, rows_B: int) -> np.ndarray:
    """Return edge weights with length exactly rows_B / 3.
    If a precomputed file exists, load and trim if necessary; otherwise generate."""
    fname = os.path.join(dir_path, f"weights_rr_{opt}.npy")
    if os.path.isfile(fname):
        w = np.load(fname)
        if w.shape[0] != rows_B // 3:
            print(f"[WARN] loaded weights length {w.shape[0]} ≠ |E_RR|={rows_B//3}; trimming …")
            w = w[: rows_B // 3]
        return w

    print(f"[INFO] pre-computed weights_rr_{opt}.npy not found — generating …")
    w = build_weights_rr(dir_path, opt, labels)
    # Enforce correct length
    if w.shape[0] < rows_B // 3:
        sys.exit("[ERROR] generated weights shorter than expected — check steps 05/06")
    if w.shape[0] > rows_B // 3:
        print("[WARN] generated weights longer than needed; extra entries will be trimmed …")
        w = w[: rows_B // 3]
    np.save(fname, w)
    print(f"[INFO] saved {w.shape[0]} weights → {fname}")
    return w

# ─────────────────────────────────────────────────────────────────────────────
# ADMM core logic
# ─────────────────────────────────────────────────────────────────────────────

def admm_gTV(B: sp.csr_matrix, v: np.ndarray, C: sp.csr_matrix, q: np.ndarray,
             p_init: np.ndarray, w_edge: np.ndarray,
             lam: float, rho: float, max_iters: int, tol: float,
             eps_reg: float = 1e-9) -> np.ndarray:
    """ADMM for graph total variation on normals.
    Variables are per-coordinate (x,y,z) for the selected vertex set."""
    n_vars = B.shape[1]
    rows_B = B.shape[0]  # 3 * |E_RR|

    # Pre-factor normal equations matrix (regularized for stability)
    A = C.T @ C + rho * (B.T @ B) + eps_reg * sp.eye(n_vars, format="csr")
    solver = spla.factorized(A.tocsc())

    # ADMM state
    p = p_init.copy()
    m = B @ p + v
    u = np.zeros_like(m)

    w_rep = np.repeat(w_edge, 3)  # (rows_B,)

    for it in range(max_iters):
        # p-update
        rhs = C.T @ q + rho * B.T @ (m - v - u)
        p = solver(rhs)

        # m-update (vectorized ℓ2 shrinkage over 3D edge blocks)
        t = B @ p + v - u
        t_rows = t.reshape(-1, 3)
        norms = np.linalg.norm(t_rows, axis=1)
        scale = np.maximum(1.0 - (lam * w_edge) / (rho * (norms + 1e-12)), 0.0)
        m_prev = m.copy()
        m = (np.repeat(scale, 3) * t)

        # Dual update
        u += B @ p + v - m

        # Residuals
        r_norm = np.linalg.norm(B @ p + v - m)
        s_norm = np.linalg.norm(rho * B.T @ (m - m_prev))

        if it % 10 == 0 or (r_norm < tol and s_norm < tol):
            print(f"iter {it:03d} | r={r_norm:.3e} | s={s_norm:.3e}")
        if r_norm < tol and s_norm < tol:
            print(f"[INFO] converged in {it} iterations")
            break
    else:
        print("[WARN] reached max-iters without full convergence")

    return p

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Step 07 — ADMM solver (gTV on normals)")
    parser.add_argument("--opt", choices=["red", "blue"], default="red",
                        help="target set being optimized")
    parser.add_argument("--lam", type=float, default=5e-2, help="lambda for gTV term")
    parser.add_argument("--rho", type=float, default=1.0, help="ADMM rho")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="stopping tolerance for primal/dual residuals")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--dir", default="intermediate",
                        help="folder containing matrices from steps 05–06")
    args = parser.parse_args()

    DIR = args.dir
    labels = np.load(os.path.join(DIR, "labels.npy"))
    set_id = 0 if args.opt == "red" else 1
    idx_opt = np.flatnonzero(labels == set_id)
    coord_opt = np.sort(np.concatenate([3 * idx_opt + d for d in range(3)]))

    print(f"[INFO] optimising {len(idx_opt)} {args.opt.upper()} vertices …")
    print(f"[INFO] variable count = {len(coord_opt):,}")

    # ─ Load matrices ───────────────────────────────────────────────────────
    B_full = load_sparse(os.path.join(DIR, f"B_{args.opt}.npz"))
    v = np.load(os.path.join(DIR, f"v_{args.opt}.npy"))
    C_full = load_sparse(os.path.join(DIR, "C.npz"))
    q = np.load(os.path.join(DIR, "q.npy"))

    B = B_full[:, coord_opt]
    C = C_full[:, coord_opt]

    # Weights (auto-generate if missing)
    w_edge = ensure_weights_rr(DIR, args.opt, labels, rows_B=B.shape[0])

    # Initial p from original coordinates
    pts = np.loadtxt(os.path.join(DIR, "points_all.xyz"))
    p_init = pts.flatten()[coord_opt]

    # ─ ADMM optimization ────────────────────────────────────────────────────
    p_opt = admm_gTV(B, v, C, q.flatten(), p_init,
                     w_edge=w_edge,
                     lam=args.lam, rho=args.rho,
                     max_iters=args.max_iters, tol=args.tol)

    # ─ Write back ───────────────────────────────────────────────────────────
    pts_out = pts.flatten()
    pts_out[coord_opt] = p_opt
    out_path = os.path.join(DIR, f"points_optim_{args.opt}.xyz")
    np.savetxt(out_path, pts_out.reshape(-1, 3), fmt="%.6f")
    print("[INFO] written", out_path)

if __name__ == "__main__":
    main()
