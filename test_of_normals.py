import numpy as np
import open3d as o3d

# ======== تنظیمات کاربر ========
points_path = "intermediate/points_all.xyz"  # مسیر فایل نقاط
k = 8  # تعداد همسایه‌ها
# ==============================

# بارگذاری نقاط
points = np.loadtxt("input/Asterix_downsampled_30pct.xyz")
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

# برآورد نرمال‌ها
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.normalize_normals()

# ساخت kNN edges
tree = o3d.geometry.KDTreeFlann(pcd)
lines = []
for i in range(len(points)):
    _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], k+1)
    for j in idx[1:]:
        if i < j:
            lines.append([i, j])

line_set = o3d.geometry.LineSet()
line_set.points = pcd.points
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))  # خاکستری

# رنگ‌دهی نقاط براساس نرمال (normalize به 0-1)
normals = np.asarray(pcd.normals)
colors = (normals - normals.min(axis=0)) / (normals.max(axis=0) - normals.min(axis=0))
pcd.colors = o3d.utility.Vector3dVector(colors)

# نمایش
o3d.visualization.draw_geometries(
    [pcd, line_set],
    window_name="k-NN Graph & Normals",
    point_show_normal=True  # خطوط کوچک نرمال‌ها
)
