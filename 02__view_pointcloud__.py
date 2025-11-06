import open3d as o3d

file_path1 = "input/Asterix_downsampled_30pct.xyz"  # مسیر فایل دقیق

pcd1 = o3d.io.read_point_cloud(file_path1, format='xyz')

print(f"[INFO] Loaded {len(pcd1.points)} points")

o3d.visualization.draw_geometries([pcd1], window_name="Point Cloud Viewer")

"""
import numpy as 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- مسیر فایل رو اینجا بذار ---
file_path = "intermediate/points_all.xyz"

# خواندن فایل XYZ
pts = np.loadtxt(file_path)
x, y, z = pts[:,0], pts[:,1], pts[:,2]

# رسم سه‌بعدی
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, c=z, cmap='viridis')  # رنگ بر اساس Z
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud Viewer (Matplotlib)')
plt.show()
"""