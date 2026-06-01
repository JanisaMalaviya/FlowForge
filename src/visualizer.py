import sys
import os
import joblib
from src.features import extract_piml_features
# Assuming you have a visualizer function in src/visualizer.py
from src.visualizer import launch_3d_visualizer 

MODEL_PATH = "models/aerodynamic_model.pkl"
CAD_DIR = "data/raw_cad/"

def main():
    if len(sys.argv) < 2:
        print("\n[!] Usage: python run_analyzer.py <your_filename.stl>")
        return

    filename = sys.argv[1]
    filepath = os.path.join(CAD_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[!] Error: File not found at {filepath}")
        return

    print(f"Reading physics and geometry from {filepath}...")
    
    # 1. Extract the new physics features
    feature_vector, point_cloud = extract_piml_features(filepath)
    
    # 2. Load the trained sklearn pipeline
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("[!] Model not found. Run train.py first to build the PIML model.")
        return
        
    # 3. Predict Cd (reshape because sklearn expects a 2D array for single predictions)
    predicted_cd = model_pipeline.predict(feature_vector.reshape(1, -1))[0]
    
    print(f"Analysis Complete. Physics-Informed Predicted Cd: {predicted_cd:.5f}")
    
    # 4. Launch PyVista
    print("Launching 3D Visualizer...")
    launch_3d_visualizer(point_cloud, predicted_cd)

if __name__ == "__main__":
    main()