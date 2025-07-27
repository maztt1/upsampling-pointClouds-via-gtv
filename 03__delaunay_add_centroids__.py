# 03__delaunay_add_centroids__.py
import numpy as np
import open3d as o3d
import os

def compute_centroids_delaunay_xyz(xyz_path: str, alpha: float | None = None,
                                   save_xyz: bool = True, out_dir: str = "output"):
    """
    from a low-res point cloud
    inserts new points at triangle centroids.

    Returns
    -------
    centroids : (M,3) ndarray
        Array of inserted points.
    o3d.geometry.TriangleMesh.
    """

    pts = np.loadtxt(xyz_path)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # --- adaptive α if not provided ---
    if alpha is None:
        dists = pcd.compute_nearest_neighbor_distance()
        alpha = 2.0 * np.mean(dists)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)          # same coords as `pts` (possibly re-ordered)
    centroids = verts[tris].mean(axis=1)       # (M,3)

    if save_xyz:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(xyz_path))[0]
        out_path = os.path.join(out_dir, f"{base}_centroids.xyz")
        np.savetxt(out_path, centroids, fmt="%.6f")
        print(f"[INFO] Saved {len(centroids)} centroids → {out_path}")

    return centroids, mesh


if __name__ == "__main__":
    low_res_xyz = "input/Asterix_downsampled_30pct.xyz"
    new_pts, surf_mesh = compute_centroids_delaunay_xyz(low_res_xyz, alpha=None)

    # --- visualise original vs new points ---
    pcd_orig = o3d.io.read_point_cloud(low_res_xyz, format='xyz')
    pcd_orig.paint_uniform_color([0, 0, 1])     # blue
    pcd_new  = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_pts))
    pcd_new.paint_uniform_color([1, 0, 0])      # red

    o3d.visualization.draw_geometries([pcd_orig, pcd_new],
        window_name="Original (Blue)  +  New Centroids (Red)")
