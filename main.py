import os
import pandas as pd
import numpy as np
from src.simulator import calculate_aerodynamic_coefficients
from src.train import train_model
from src.predict import analyze_new_file

DATASET_FILE = "data/dataset.csv"
MODEL_FILE = "models/aerodynamic_model.pkl"

# --- SYSTEM MODE ---
# Options: "train" or "predict"
MODE = "train" 
# -------------------

def generate_wing_geometry():
    """Parametric digital wind tunnel generation for AI training."""
    chord = np.random.uniform(0.5, 2.0)
    span = np.random.uniform(2.0, 10.0)
    thickness = np.random.uniform(0.05, 0.2) * chord 
    
    x = np.random.uniform(0, chord, 1000)
    y = np.random.uniform(0, span, 1000)
    z_limit = (thickness/2) * (1 - (x/chord)**2)
    z = np.random.uniform(-z_limit, z_limit, 1000)
    
    return np.stack((x, y, z), axis=-1).flatten()

def main():
    os.makedirs('data/raw_cad', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    if MODE == "train":
        print("Generating 500 high-fidelity physics models...")
        wing_data = []
        for _ in range(500):
            geometry = generate_wing_geometry()
            cd = calculate_aerodynamic_coefficients(geometry)
            wing_data.append(np.append(geometry, cd))
        
        cols = [f'p{i}{ax}' for i in range(1000) for ax in ('x','y','z')] + ['Cd']
        df = pd.DataFrame(wing_data, columns=cols)
        df.to_csv(DATASET_FILE, index=False)
        
        print("Feeding data to the Random Forest Regressor...")
        train_model(DATASET_FILE, MODEL_FILE)
        print("Training Sequence Terminated: Aerodynamic Inference Model initialized and saved.")

    elif MODE == "predict":
        # Put your exact Autodesk Fusion STL filename here
        test_file = "data/raw_cad/my_fusion_wing.stl"
        
        if not os.path.exists(test_file):
            print(f"ERROR: Cannot find {test_file}. Did you export it to the right folder?")
            return
            
        analyze_new_file(test_file, MODEL_FILE)

if __name__ == "__main__":
    main()