import trimesh
import numpy as np

def cad_to_point_cloud(stl_path, num_points=1000):
    """Loads STL from Fusion/AutoCAD and normalizes its orientation."""
    try:
        mesh = trimesh.load(stl_path)
        
        # Align longest dimension to the X-axis (Chord-wise airflow)
        mesh.apply_transform(trimesh.geometry.align_vectors([1,0,0], mesh.principal_inertia_vectors[0]))
        
        # Center the model at coordinate (0,0,0)
        mesh.vertices -= mesh.center_mass
        
        # Extract the uniform surface point cloud
        point_cloud, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        return point_cloud.flatten()
    except Exception as e:
        print(f"Error: Could not process {stl_path}.")
        print(f"Details: {e}")
        return None