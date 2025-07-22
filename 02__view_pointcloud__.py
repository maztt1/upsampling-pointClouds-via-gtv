import open3d as o3d

file_path1 = "input/Asterix_downsampled_30pct.xyz"  # مسیر فایل دقیق

pcd1 = o3d.io.read_point_cloud(file_path1, format='xyz')

print(f"[INFO] Loaded {len(pcd1.points)} points")

o3d.visualization.draw_geometries([pcd1], window_name="Point Cloud Viewer")





