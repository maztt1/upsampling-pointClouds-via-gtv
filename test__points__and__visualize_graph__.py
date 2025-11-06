import open3d as o3d
import numpy as np

# بارگذاری نقاط و یال‌ها
points = np.loadtxt("intermediate/points_all.xyz")
edges  = np.load("intermediate/edges.npy")

# تعریف LineSet برای گراف
lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(points)
lines.lines = o3d.utility.Vector2iVector(edges)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
o3d.visualization.draw_geometries(
    [pcd, lines],
    window_name="Points + Graph",
    width=1000, height=800
)
