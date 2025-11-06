import numpy as np
import open3d as o3d

# -------- تنظیم مسیر فایل‌ها (تو اینجا مسیر خودتو بذار) --------
points_path = "intermediate/points_all.xyz"     # فایل نقاط
labels_path = "intermediate/labels.npy"         # فایل لیبل (0 = قرمز، 1 = آبی)
edges_path  = "intermediate/edges_final.npy"    # فایل یال‌ها
# ---------------------------------------------------------------

# بارگذاری داده‌ها
points = np.loadtxt(points_path)
labels = np.load(labels_path)
edges  = np.load(edges_path)

# ساخت PointCloud با رنگ‌ها
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
colors = np.zeros((len(points), 3))
colors[labels == 0] = [1, 0, 0]   # قرمز
colors[labels == 1] = [0, 0, 1]   # آبی
pcd.colors = o3d.utility.Vector3dVector(colors)

# ساخت LineSet برای یال‌ها
line_set = o3d.geometry.LineSet()
line_set.points = pcd.points
line_set.lines = o3d.utility.Vector2iVector(edges)
line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(edges))  # خاکستری

# نمایش در Open3D
o3d.visualization.draw_geometries(
    [pcd, line_set],
    window_name="Bipartite Graph (Red & Blue Nodes + Edges)"
)

import numpy as np

labels = np.load("intermediate/labels.npy")
edges_final = np.load("intermediate/edges_final.npy")
removed_edges = np.load("intermediate/removed_edges.npy")

red_points = np.sum(labels == 0)
blue_points = np.sum(labels == 1)
final_edges = len(edges_final)
removed_edges_count = len(removed_edges)

print(f"Red points: {red_points}")
print(f"Blue points: {blue_points}")
print(f"Final edges: {final_edges}")
print(f"Removed edges: {removed_edges_count}")
