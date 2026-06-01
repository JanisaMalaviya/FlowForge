import trimesh
import numpy as np

def extract_piml_features(filepath, num_points=3000):
    """
    Physics-Informed Machine Learning (PIML) Feature Extractor.
    Extracts absolute aerodynamic theories + normalized geometric points.
    """
    # 1. Load watertight mesh
    mesh = trimesh.load(filepath, force='mesh')
    
    # --- A. THE PHYSICS VECTOR ---
    # Calculates Form Drag, Skin Friction, and Wake generation parameters
    
    length, width, height = mesh.extents
    surface_area = mesh.area  # Skin Friction metric
    volume = mesh.volume
    
    # Frontal Area (Assuming airflow along X-axis, hitting the Y-Z plane)
    frontal_area = width * height 
    
    # Solidity (Blockage Ratio)
    bb_volume = length * width * height
    solidity = volume / bb_volume if bb_volume > 0 else 0
    
    # Fineness Ratio (Aerodynamic slenderness)
    eff_diameter = 2 * np.sqrt(frontal_area / np.pi) if frontal_area > 0 else 0
    fineness_ratio = length / eff_diameter if eff_diameter > 0 else 0
    
    # Compile physics theories into an 8-feature array
    physics_features = np.array([
        length, width, height, 
        surface_area, volume, frontal_area, 
        solidity, fineness_ratio
    ])
    
    # --- B. THE GEOMETRY VECTOR ---
    # Sample points to capture the actual shape/curvature (like airfoils)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Center the points at the origin (0,0,0) for the visualizer
    centroid = np.mean(points, axis=0)
    normalized_points = points - centroid
    
    # Flatten the 3000x3 point cloud into a 1D array (9000 features) for sklearn
    flattened_points = normalized_points.flatten()
    
    # --- C. FUSION ---
    # Combine Physics (8 features) + Geometry (9000 features)
    final_feature_vector = np.hstack((physics_features, flattened_points))
    
    return final_feature_vector, normalized_points