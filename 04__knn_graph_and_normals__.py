# 04__knn_graph_and_normals__.py
import numpy as np
import open3d as o3d
import pickle, os

def build_knn_graph_xyz(xyz_combined: np.ndarray, k: int = 8):
    #build point cloud from all points(low + centroids)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_combined))
    #estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    #normalize normals
    pcd.normalize_normals()

    # وزن‌ها: exp( -||p_i-p_j||² / σ_p² ) * cos² θ
    tree = o3d.geometry.KDTreeFlann(pcd)
    # average distance for each point to its k nearest neighbors
    σ_p = np.mean(pcd.compute_nearest_neighbor_distance())
    edges, weights = [], []
    normals = np.asarray(pcd.normals)
    points  = np.asarray(pcd.points)

    for i in range(len(points)):
        _, idx, dist2 = tree.search_knn_vector_3d(points[i], k+1)
        for j, d2 in zip(idx[1:], dist2[1:]):
            cosθ = np.dot(normals[i], normals[j])
            w = np.exp(-d2 / σ_p**2) * (cosθ**2)
            edges.append((i, j))
            weights.append(w)

    return pcd, np.array(edges), np.array(weights)

if __name__ == "__main__":
    xyz_low  = np.loadtxt("input/Asterix_downsampled_30pct.xyz")
    xyz_cent = np.loadtxt("intermediate/Asterix_downsampled_30pct_centroids.xyz")
    xyz_all  = np.vstack((xyz_low, xyz_cent))

    pcd, E, W = build_knn_graph_xyz(xyz_all, k=8)

    # save point cloud with normals
    os.makedirs("intermediate", exist_ok=True)
    np.savetxt("intermediate/points_all.xyz", xyz_all, fmt="%.6f")
    np.save("intermediate/edges.npy",    E)
    np.save("intermediate/weights.npy",  W)
    np.save("intermediate/normals.npy",  np.asarray(pcd.normals))

    # --- تست طول نرمال‌ها ---
    norms = np.linalg.norm(np.asarray(pcd.normals), axis=1)
    print(f"[CHECK] normal magnitudes: mean={np.mean(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}")
    
    o3d.visualization.draw_geometries(
        [pcd],
        point_show_normal=True,
        window_name="Visualize normals"
    )

    print(f"[INFO] graph saved: {len(E)} edges on {len(xyz_all)} points")
