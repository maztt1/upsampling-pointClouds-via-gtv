import trimesh
import numpy as np
import os

def extract_and_downsample(input_ply_path, output_dir, downsample_ratio=0.3):

    # Load mesh
    mesh = trimesh.load(input_ply_path)
    vertices = mesh.vertices
    num_total = len(vertices)

    print(f"[INFO] Loaded {num_total} points from: {input_ply_path}")

    # Save all vertices (full point cloud)
    base_name = os.path.splitext(os.path.basename(input_ply_path))[0]
    full_path = os.path.join(output_dir, f"{base_name}_full.xyz")
    np.savetxt(full_path, vertices, fmt="%.6f")
    print(f"[INFO] Full point cloud saved to: {full_path}")

    # Downsample
    num_sample = int(num_total * downsample_ratio)
    indices = np.random.choice(num_total, num_sample, replace=False)
    downsampled = vertices[indices]

    downsampled_path = os.path.join(output_dir, f"{base_name}_downsampled_{int(downsample_ratio * 100)}pct.xyz")
    np.savetxt(downsampled_path, downsampled, fmt="%.6f")
    print(f"[INFO] Downsampled ({downsample_ratio*100:.0f}%) saved to: {downsampled_path}")

    # after downsampling (for the step 06)
    indices = np.random.choice(num_total, num_sample, replace=False)
    ...
    np.save(os.path.join(output_dir, "q_indices.npy"), indices)

    return full_path, downsampled_path

if __name__ == "__main__":
    input_file = r"C:\Users\Asus\PycharmProjects\PythonProject2\data\4arms_monstre.ply"
    output_folder = "output"
    downsample_ratio = 0.3
    os.makedirs(output_folder, exist_ok=True)
    extract_and_downsample(input_file, output_folder, downsample_ratio)
