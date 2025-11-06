import open3d as o3d
import numpy as np

# بارگذاری نقاط و یال‌ها
points = np.loadtxt("intermediate/points_all.xyz")
edges  = np.load("intermediate/edges.npy")

# تعریف LineSet برای گراف
lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(points)
lines.lines = o3d.utility.Vector2iVector(edges)

# نمایش
o3d.visualization.draw_geometries(
    [lines],
    window_name="k-NN Graph (lines only)",
    width=1000, height=800
)
