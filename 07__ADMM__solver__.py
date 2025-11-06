#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 07 — ADMM solver for Graph-TV on surface normals (v3.0)
- Solves per-color (RED/BLUE) using B_{opt}.npz, v_{opt}.npy, C.npz, q.npy.
- Stable by default: weight normalisation + adaptive rho + tunable eps_reg.
Outputs: intermediate/points_optim_{red|blue}.xyz
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────
def load_sparse(path: str) -> sp.csr_matrix:
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] file not found: {path}")
    return sp.load_npz(path).tocsr()

def build_weights_rr(dir_path: str, opt: str, labels: np.ndarray) -> np.ndarray:
    edges   = np.load(os.path.join(dir_path, "edges_final.npy"))          # (E,2)
    weights = np.load(os.path.join(dir_path, "weights_final.npy"))        # (E,)
    iso_p   = os.path.join(dir_path, "isolated_nodes.npy")
    isolated = np.load(iso_p) if os.path.isfile(iso_p) else np.array([], dtype=int)

    mask_iso = np.zeros(labels.shape[0], dtype=bool)
    mask_iso[isolated] = True
    set_id = 0 if opt == "red" else 1
    mask_set = labels == set_id

    mask_rr = (
        mask_set[edges[:, 0]] & mask_set[edges[:, 1]] &
        ~mask_iso[edges[:, 0]] & ~mask_iso[edges[:, 1]]
    )
    return weights[mask_rr]

def ensure_weights_rr(dir_path: str, opt: str, labels: np.ndarray, rows_B: int) -> np.ndarray:
    """Return w_edge of length rows_B/3 (normalised by median)."""
    need = rows_B // 3
    fname = os.path.join(dir_path, f"weights_rr_{opt}.npy")
    if os.path.isfile(fname):
        w = np.load(fname)
    else:
        print(f"[INFO] pre-computed weights_rr_{opt}.npy not found — generating …")
        w = build_weights_rr(dir_path, opt, labels)
        np.save(fname, w)
        print(f"[INFO] saved {w.shape[0]} weights -> {fname}")

    if w.shape[0] < need:
        sys.exit(f"[ERROR] weights for {opt} too short: {w.shape[0]} < expected {need}")
    if w.shape[0] > need:
        print(f"[WARN] weights for {opt} longer than needed ({w.shape[0]}>{need}); trimming")
        w = w[:need]

    # normalise (important for stability)
    w = w.astype(np.float64, copy=False)
    med = float(np.median(w)) if w.size else 1.0
    if med <= 0: med = 1.0
    w /= med
    return w

# ────────────────────────────────────────────────────────────────────────────
# ADMM core (with adaptive rho)
# ────────────────────────────────────────────────────────────────────────────
def admm_gTV(B: sp.csr_matrix, v: np.ndarray, C: sp.csr_matrix, q: np.ndarray,
             p_init: np.ndarray, w_edge: np.ndarray,
             lam: float, rho: float, max_iters: int, tol: float,
             eps_reg: float = 1e-6) -> np.ndarray:
    """
    m = Bp + v, TV on edge-wise 3D blocks of m with weights w_edge.
    Adaptive-ρ per Boyd et al.: balances primal/dual residuals and refactorises A.
    """
    n_vars = B.shape[1]
    rows_B = B.shape[0]
    if rows_B != v.shape[0]:
        sys.exit(f"[ERROR] B rows ({rows_B}) != len(v) ({len(v)})")
    if rows_B % 3 != 0:
        sys.exit("[ERROR] B rows must be a multiple of 3")

    # helper: factorise A for a given rho value
    def make_solver(rho_val: float):
        A = (C.T @ C) + (rho_val * (B.T @ B)) + (eps_reg * sp.eye(n_vars, format="csr"))
        return spla.factorized(A.tocsc())

    rho_cur = float(rho)
    solver = make_solver(rho_cur)

    # ADMM state
    p = p_init.copy()
    m = B @ p + v
    u = np.zeros_like(m)

    # residual balancing params
    mu, tau_inc, tau_dec = 10.0, 2.0, 2.0

    for it in range(max_iters + 1):
        # p-update
        rhs = C.T @ q + rho_cur * B.T @ (m - v - u)
        p = solver(rhs)

        # m-update (3D block soft-thresholding)
        t = B @ p + v - u
        t_blocks = t.reshape(-1, 3)
        norms = np.linalg.norm(t_blocks, axis=1) + 1e-12
        scale = np.maximum(1.0 - (lam * w_edge) / (rho_cur * norms), 0.0)
        m_prev = m
        m = (np.repeat(scale, 3) * t)

        # dual update
        u += (B @ p + v - m)

        # residuals
        r_norm = np.linalg.norm(B @ p + v - m)
        s_norm = np.linalg.norm(rho_cur * B.T @ (m - m_prev))

        if it % 10 == 0 or it == 0 or (r_norm < tol and s_norm < tol):
            print(f"iter {it:03d} | r={r_norm:.3e} | s={s_norm:.3e} | rho={rho_cur:.3g}")

        # adaptive rho
        if r_norm > mu * s_norm:
            rho_cur *= tau_inc
            solver = make_solver(rho_cur)
        elif s_norm > mu * r_norm and rho_cur > 1e-12:
            rho_cur /= tau_dec
            solver = make_solver(rho_cur)

        if r_norm < tol and s_norm < tol:
            print(f"[INFO] converged in {it} iterations")
            break
    else:
        print("[WARN] reached max-iters with full convergence")

    return p

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Step 07 — ADMM solver (gTV on normals)")
    parser.add_argument("--opt", choices=["red", "blue"], default="red", help="which set to optimise")
    parser.add_argument("--lam", type=float, default=5e-2, help="lambda for gTV term")
    parser.add_argument("--rho", type=float, default=1.0, help="initial ADMM rho")
    parser.add_argument("--tol", type=float, default=1e-4, help="stopping tolerance")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--eps-reg", type=float, default=1e-6, help="Tikhonov regulariser on p-update system")
    parser.add_argument("--dir", default="intermediate", help="folder containing matrices from steps 05–06")
    args = parser.parse_args()

    DIR = args.dir
    labels = np.load(os.path.join(DIR, "labels.npy"))
    set_id = 0 if args.opt == "red" else 1
    idx_opt = np.flatnonzero(labels == set_id)
    coord_opt = np.sort(np.concatenate([3*idx_opt + d for d in range(3)]))

    print(f"[INFO] optimising {len(idx_opt)} {args.opt.upper()} vertices …")
    print(f"[INFO] variable count = {len(coord_opt):,}")

    # Load matrices
    B_full = load_sparse(os.path.join(DIR, f"B_{args.opt}.npz"))
    v = np.load(os.path.join(DIR, f"v_{args.opt}.npy"))
    C_full = load_sparse(os.path.join(DIR, "C.npz"))
    q = np.load(os.path.join(DIR, "q.npy"))
    if B_full.shape[1] <= coord_opt.max():
        sys.exit(f"[ERROR] B has {B_full.shape[1]} cols but max coord index is {coord_opt.max()}")
    if C_full.shape[1] <= coord_opt.max():
        sys.exit(f"[ERROR] C has {C_full.shape[1]} cols but max coord index is {coord_opt.max()}")

    # Column-slice to chosen set
    B = B_full[:, coord_opt]
    C = C_full[:, coord_opt]

    # Weights (normalised)
    w_edge = ensure_weights_rr(DIR, args.opt, labels, rows_B=B.shape[0])

    # Initial p from original coords
    pts = np.loadtxt(os.path.join(DIR, "points_all.xyz"))
    p_init = pts.flatten()[coord_opt]

    # Solve
    p_opt = admm_gTV(B, v, C, q.flatten(), p_init,
                     w_edge=w_edge,
                     lam=args.lam, rho=args.rho,
                     max_iters=args.max_iters, tol=args.tol,
                     eps_reg=args.eps_reg)

    # Write back
    pts_out = pts.flatten()
    pts_out[coord_opt] = p_opt
    out_path = os.path.join(DIR, f"points_optim_{args.opt}.xyz")
    np.savetxt(out_path, pts_out.reshape(-1, 3), fmt="%.6f")
    print("[INFO] written", out_path)

if __name__ == "__main__":
    main()
