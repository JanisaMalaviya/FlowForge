import numpy as np

def calculate_aerodynamic_coefficients(flattened_points, velocity=50.0):
    """Calculates Cd based on Skin Friction, Form Factor, and Induced Drag."""
    points = flattened_points.reshape(-1, 3)
    
    chord = np.max(points[:, 0]) - np.min(points[:, 0])
    span = np.max(points[:, 1]) - np.min(points[:, 1])
    thickness = np.max(points[:, 2]) - np.min(points[:, 2])
    
    if chord <= 0 or span <= 0: return 0.05
    
    aspect_ratio = (span**2) / (span * chord)
    thickness_ratio = thickness / chord

    # Standard Atmosphere
    rho = 1.225; mu = 1.81e-5; nu = mu / rho
    re = (velocity * chord) / nu

    # Parasitic Drag
    cf = 0.455 / (np.log10(re) ** 2.58) if re > 5e5 else 1.328 / np.sqrt(re)
    ff = 1 + 2.0 * thickness_ratio + 100 * (thickness_ratio**4)
    cd_profile = cf * ff * 2.0 

    # Induced Drag (Assuming 5 deg AoA)
    cl = 2 * np.pi * np.radians(5) 
    cd_induced = (cl**2) / (np.pi * aspect_ratio * 0.85)

    return float(cd_profile + cd_induced)