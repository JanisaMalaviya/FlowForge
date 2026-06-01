import trimesh
import numpy as np
import joblib

def extract_physics_and_geometry(filepath, num_points=3000):
    """Extracts absolute aerodynamic parameters for ML, and point cloud for visualization."""
    mesh = trimesh.load(filepath, force='mesh')
    
    # 1. Engineering Physics Features (8 dimensions)
    length, width, height = mesh.extents
    surface_area = mesh.area  
    volume = mesh.volume
    frontal_area = width * height  
    
    bb_volume = length * width * height
    solidity = volume / bb_volume if bb_volume > 0 else 0
    
    eff_diameter = 2 * np.sqrt(frontal_area / np.pi) if frontal_area > 0 else 0
    fineness_ratio = length / eff_diameter if eff_diameter > 0 else 0
    
    # This is the clean, 8-feature vector for the ML model
    physics_features = np.array([
        length, width, height, 
        surface_area, volume, frontal_area, 
        solidity, fineness_ratio
    ])
    
    # 2. Geometry Point Cloud (Kept purely for PyVista rendering)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    centroid = np.mean(points, axis=0)
    normalized_points = points - centroid
    
    return physics_features, normalized_points

def analyze_new_file(filepath, model_path):
    """Predicts Cd using only pure engineering parameters."""
    physics_features, point_cloud = extract_physics_and_geometry(filepath)
    
    model = joblib.load(model_path)
    
    # Predict using the clean 8-feature shape
    predicted_cd = model.predict(physics_features.reshape(1, -1))[0]
    
    return predicted_cd, point_cloud