#!/usr/bin/env python3
"""
Step 06 — Prepare ADMM matrices (v3: correct constraints)
- B_{opt}.npz: 3*|E_set| x 3*N_total
- v_{opt}.npy: len = 3*|E_set|
- C.npz, q.npy: constrain ORIGINAL low-res points (first M rows), where
  M is read from --low-res .xyz (default: input/low_res.xyz if present).
"""
import argparse, os, sys
import numpy as np
import scipy.sparse as sp

def load_labels_edges(dir_path: str):
    labels = np.load(os.path.join(dir_path, "labels.npy"))
    edges  = np.load(os.path.join(dir_path, "edges_final.npy")).astype(np.int64)
    iso_p  = os.path.join(dir_path, "isolated_nodes.npy")
    isolated = np.load(iso_p) if os.path.isfile(iso_p) else np.array([], dtype=np.int64)
    return labels, edges, isolated

def build_Bv(points: np.ndarray, edges: np.ndarray,
             labels: np.ndarray, set_id: int, isolated: np.ndarray):
    mask_iso = np.zeros(labels.shape[0], dtype=bool)
    mask_iso[isolated] = True
    sel = (labels[edges[:,0]] == set_id) & (labels[edges[:,1]] == set_id) \
          & (~mask_iso[edges[:,0]]) & (~mask_iso[edges[:,1]])
    E_sel = edges[sel]
    N_total = points.shape[0]
    rows, cols, data, v_list = [], [], [], []
    for e,(i,j) in enumerate(E_sel):
        base = 3*e
        # simple linear proxy: n_i - n_j ≈ p_i - p_j
        for d in range(3):
            rows.append(base+d); cols.append(3*i+d); data.append(+1.0)
            rows.append(base+d); cols.append(3*j+d); data.append(-1.0)
        v_list.extend(list(points[i] - points[j]))
    B = sp.csr_matrix((data,(rows,cols)), shape=(3*len(E_sel), 3*N_total))
    v = np.asarray(v_list, dtype=np.float64)
    return B, v, E_sel

def build_Cq_fix_first_M(points_all: np.ndarray, M: int):
    """Pin the first M global vertices to their current coords."""
    N_total = points_all.shape[0]
    if M <= 0 or M > N_total:
        sys.exit(f"[ERROR] invalid M={M} for C (N_total={N_total})")
    rows, cols, data, q = [], [], [], []
    for r, gidx in enumerate(range(M)):
        for d in range(3):
            rows.append(3*r+d); cols.append(3*gidx+d); data.append(1.0)
            q.append(points_all[gidx, d])
    C = sp.csr_matrix((data,(rows,cols)), shape=(3*M, 3*N_total))
    return C, np.asarray(q, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser("Step 06 — prepare ADMM matrices (v3)")
    ap.add_argument("--opt", choices=["red","blue"], required=True,
                    help="which set to prepare")
    ap.add_argument("--dir", default="intermediate",
                    help="folder with points_all.xyz, labels.npy, edges_final.npy")
    ap.add_argument("--low-res", default=None,
                    help="path to low-res .xyz to count M originals (default tries input/*.xyz)")
    args = ap.parse_args()

    set_id = 0 if args.opt=="red" else 1
    pts_all = np.loadtxt(os.path.join(args.dir, "points_all.xyz"))
    labels, edges, isolated = load_labels_edges(args.dir)

    # Determine M (original low-res count)
    if args.low_res is not None:
        low_path = args.low_res

    else:
        # try common defaults; adjust if your path differs
        # 08__merge_and_evaluate__ used these names, so try them:
        candidates = [
            "input/Asterix_downsampled_30pct.xyz",
            "input/low_res.xyz",
            "input/low.xyz"
        ]
        low_path = next((p for p in candidates if os.path.isfile(p)), None)
    if low_path is None:
        sys.exit("[ERROR] --low-res not given and no default input/*low*.xyz found")
    M = np.loadtxt(low_path).shape[0]
    print(f"[INFO] using M={M} original vertices from {low_path}")

    # Build and save
    B,v,E_sel = build_Bv(pts_all, edges, labels, set_id, isolated)
    sp.save_npz(os.path.join(args.dir, f"B_{args.opt}.npz"), B)
    np.save(os.path.join(args.dir, f"v_{args.opt}.npy"), v)

    C,q = build_Cq_fix_first_M(pts_all, M)
    sp.save_npz(os.path.join(args.dir, "C.npz"), C)
    np.save(os.path.join(args.dir, "q.npy"), q)

    print(f"[INFO] B_{args.opt}.npz  shape {B.shape}")
    print(f"[INFO] v_{args.opt}.npy  len {len(v)}")
    print(f"[INFO] C.npz            shape {C.shape}  (M={M})")
    print(f"[INFO] q.npy            len {len(q)}")
    print(f"[INFO] edges used       {len(E_sel)}")

if __name__ == "__main__":
    main()
