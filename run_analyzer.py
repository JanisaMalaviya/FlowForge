import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.predict import analyze_new_file
except ImportError:
    print("Error: Unable to import 'analyze_new_file' from 'src.predict'.")
    print("Ensure terminal working path is set to the project root directory.")
    sys.exit(1)

MODEL_PATH = "models/aerodynamic_model.pkl"
CAD_DIR = "data/raw_cad/"

def main():
    print("-" * 60)
    print("AeroML Aerodynamic Analyzer Pipeline")
    print("-" * 60)

    if len(sys.argv) < 2:
        print("\nError: Target filename parameter missing.")
        print("Usage: python run_analyzer.py \"filename.stl\"")
        return

    target_filename = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Trained model file not found at '{MODEL_PATH}'.")
        return

    filepath = os.path.join(CAD_DIR, target_filename)
    if not os.path.exists(filepath):
        print(f"\nError: Specified file not found: '{filepath}'")
        return

    try:
        print(f"\nExtracting physical properties from: {target_filename}...")
        predicted_cd, point_cloud = analyze_new_file(filepath, MODEL_PATH)
        
        print("\nPipeline Execution Successful")
        print(f"Predicted Drag Coefficient (Cd): {predicted_cd:.5f}")
        print("Initializing interactive 3D flow visualizer context...")
        print("-" * 60)
        
        if point_cloud is not None:
            try:
                import pyvista as pv
                import numpy as np
                print("Starting native PyVista heatmap render engine...")
                
                if isinstance(point_cloud, str) and os.path.exists(point_cloud):
                    mesh = pv.read(point_cloud)
                else:
                    mesh = point_cloud
                
                # --- HEATMAP DISTRIBUTION LOGIC ---
                # To map gradients, we extract coordinates to act as scalars
                # If point_cloud is a PolyData object, we fetch its point vertices
                points = mesh.points if hasattr(mesh, 'points') else np.array(mesh)
                
                # Map scalars based on spatial point distributions (e.g., Z-axis deflection height)
                scalars = points[:, 2] 
                
                plotter = pv.Plotter(title="AeroML Flow Distribution Heatmap")
                
                # Add the mesh with an explicit colormap and scalar array assignment
                plotter.add_mesh(
                    mesh, 
                    scalars=scalars, 
                    cmap="jet",          # Classic aero heatmap color scheme (Blue to Red)
                    point_size=4.0, 
                    render_points_as_spheres=True,
                    scalar_bar_args={"title": "Aerodynamic Pressure Delta / Scalar Scale"},
                    show_edges=False
                )
                
                plotter.add_axes()
                plotter.show()
                
            except Exception as vis_err:
                print(f"PyVista Heatmap Render Warning: {vis_err}")
        
    except Exception as e:
        print(f"\nExecution Failure: An error occurred within the engine: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()