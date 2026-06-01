**FlowForge: AI-driven 3D CAD aerodynamic simulation.**

FlowForge is an end-to-end machine learning pipeline designed to bridge the gap between structural 3D CAD design and real-time aerodynamic estimation. As a project bridging motorsport engineering and data science, FlowForge enables engineers to parse raw 3D mesh geometry (`.stl` files) and predict drag coefficients ($C_d$) within seconds, bypassing long, compute-heavy CFD sessions.

##  Key Features:

* **Geometric Feature Extraction:** Translates complex 3D mesh vertices into high-value physical aerodynamic feature sets.
* **Predictive Pipeline:** Utilizes a trained Ridge regression model to estimate drag coefficients ($C_d$) for diverse geometric profiles—from aerospace structures to complex open-wheel F1 chassis.
* **Native 3D Visualization:** Integrated PyVista rendering engine that generates interactive spatial heatmaps, visualizing pressure delta distributions across 3D coordinates.
* **Lightweight Architecture:** Optimized for efficiency by focusing on structural feature engineering over brute-force mesh analysis.

##  Project Structure:

```text
FlowForge/
├── data/           # Training datasets and raw CAD models
├── models/         # Pre-trained ML regression models
├── src/            # Core feature extraction and physics logic
├── main.py         # Entry point for pipeline execution
├── run_analyzer.py # Analysis script for individual CAD files
└── requirements.txt
