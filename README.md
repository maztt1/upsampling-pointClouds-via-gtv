# Upsampling via GTV

**3D Point Cloud Super-Resolution via Graph Total Variation on Surface Normals**
Implementation of the method described in the paper *“3D Point Cloud Super-Resolution via Graph Total Variation on Surface Normals.”*
This repository reproduces the full pipeline for upsampling sparse 3D point clouds using Python, Open3D, and SciPy.

---

## 🔧 Dependencies

* Python ≥ 3.9
* `numpy`, `scipy`, `open3d`, `trimesh`, `matplotlib`
  Install dependencies:

```bash
pip install numpy scipy open3d trimesh matplotlib
```

---

## 📁 Repository Structure

| File                             | Description                                                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **01_extract_and_downsample.py** | Loads a `.ply` mesh, extracts vertices, and creates a random downsampled point cloud.                                |
| **02_visualize_xyz.py**          | Visualizes `.xyz` point clouds using Open3D or Matplotlib.                                                           |
| **03_delaunay_add_centroids.py** | Generates new points at triangle centroids using Delaunay triangulation to densify the input cloud.                  |
| **04_knn_graph_and_normals.py**  | Builds a *k-nearest neighbors* (kNN) graph, estimates surface normals, and computes edge weights.                    |
| **06_prepare_admm_matrices.py**  | Prepares the ADMM optimization matrices (B, v, C, q).                                                                |
| **07_admm_solver.py**            | Runs the ADMM optimization for Graph Total Variation (gTV) on surface normals.                                       |
| **08_merge_and_evaluate.py**     | Merges optimized point sets, computes Chamfer/Hausdorff distances, and saves the final cloud.                        |
| **09_run_red_and_blue.py**       | Automates the entire pipeline (06→07→merge), applies multi-stage outlier filtering, and saves final colored results. |

---

## ▶️ Execution Order

Recommended processing flow:

```bash
python 01_extract_and_downsample.py
python 02_visualize_xyz.py
python 03_delaunay_add_centroids.py
python 04_knn_graph_and_normals.py
python 06_prepare_admm_matrices.py --opt red --low-res input/Asterix_downsampled_30pct.xyz
python 07_admm_solver.py --opt red
python 08_merge_and_evaluate.py
```

Or run the entire automated version:

```bash
python 09_run_red_and_blue.py --preset D --low-res input/Asterix_downsampled_30pct.xyz
```

---

## ⚙️ Key Parameters

### 03_delaunay_add_centroids.py

| Flag      | Description                             | Default       | Range/Notes                         |
| --------- | --------------------------------------- | ------------- | ----------------------------------- |
| `alpha`   | Alpha value for Delaunay reconstruction | Auto-computed | Typical: 1.5–3.0 × mean NN distance |
| `out_dir` | Output directory                        | `output`      | Must exist or will be created       |

Example:

```bash
python 03_delaunay_add_centroids.py --alpha 2.5
```

---

### 04_knn_graph_and_normals.py

| Flag | Description                 | Default | Range/Notes                     |
| ---- | --------------------------- | ------- | ------------------------------- |
| `k`  | Number of nearest neighbors | 8       | 5–20 depending on cloud density |

This step saves:

```
intermediate/points_all.xyz
intermediate/edges.npy
intermediate/weights.npy
intermediate/normals.npy
```

---

### 06_prepare_admm_matrices.py

| Flag        | Description                            | Example                                         | Notes                           |
| ----------- | -------------------------------------- | ----------------------------------------------- | ------------------------------- |
| `--opt`     | Select color set (`red` or `blue`)     | `--opt red`                                     | Run both sets separately        |
| `--dir`     | Directory containing intermediate data | `--dir intermediate`                            | Must match previous step output |
| `--low-res` | Path to low-res input cloud            | `--low-res input/Asterix_downsampled_30pct.xyz` | Used to fix constraints         |

Output:

```
intermediate/B_red.npz, v_red.npy, C.npz, q.npy
```

---

### 07_admm_solver.py

| Flag          | Description               | Default | Typical Range              |
| ------------- | ------------------------- | ------- | -------------------------- |
| `--opt`       | Optimize color set        | `red`   | `red` / `blue`             |
| `--lam`       | Regularization weight     | `5e-2`  | 1e-3 to 1e-1               |
| `--rho`       | Initial ADMM penalty      | `1.0`   | 0.5–50                     |
| `--tol`       | Convergence tolerance     | `1e-4`  | 1e-3 to 1e-6               |
| `--max-iters` | Maximum iterations        | 200     | Adjust if slow convergence |
| `--eps-reg`   | Regularization for solver | 1e-6    | Stabilizes inversion       |

Example:

```bash
python 07_admm_solver.py --opt red --lam 0.05 --rho 1.0 --tol 1e-4 --max-iters 150
```

---

### 09_run_red_and_blue.py

Automates both optimization sets and post-filtering.

| Flag                    | Description                       | Default  | Range/Notes                             |
| ----------------------- | --------------------------------- | -------- | --------------------------------------- |
| `--preset`              | Parameter preset (`D` or `E`)     | Required | `D` = more smoothing, `E` = more detail |
| `--abs-max`             | Removes extreme coordinates       | `None`   | e.g., `--abs-max 5.0`                   |
| `--radius-keep-pct`     | Keep percentile for radius filter | 99.9     | 95–100                                  |
| `--knn-k`               | k for kNN filter                  | 8        | 5–15                                    |
| `--knn-keep-pct`        | Keep percentile for kNN filter    | 99.9     | 90–100                                  |
| `--iter-outlier-passes` | Number of passes                  | 2        | 1–3                                     |

Example full command:

```bash
python 09_run_red_and_blue.py --preset D --low-res input/Asterix_downsampled_30pct.xyz --abs-max 4.5 --iter-outlier-passes 2
```

---

## 📊 Output Files

| File                                     | Description                              |
| ---------------------------------------- | ---------------------------------------- |
| `output/pointcloud_superres.xyz`         | Final reconstructed dense cloud          |
| `output/pointcloud_superres_colored.ply` | Colored visualization (red/blue sets)    |
| `intermediate/*`                         | Step-wise matrices and temporary results |

---

## 🧠 Citation

If you use or adapt this code, please cite:

> [Original paper] *3D Point Cloud Super-Resolution via Graph Total Variation on Surface Normals.*

---

## 🧩 Author

**Mohamad Ahmadzadeh**
Master’s Student, Friedrich-Alexander-Universität Erlangen-Nürnberg
HiWi Research Assistant — 3D Atom Probe Tomography Reconstruction
📧 [[mohamad.ahmadzadeh@fau.de](mailto:mohamad.ahmadzadeh@fau.de)]
