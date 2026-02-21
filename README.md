# ML-Final-Project
The goal is to analyze how different environmental states and opponent strategies affect an NPC's decision-making. The project focuses not only on algorithmic accuracy but also on user experience through Creative Reporting, which includes a dynamic web interface.

Technologies Used:
Core: Python & JAX (for high-performance computing and automatic differentiation).
Data Science: Pandas & NumPy (manual preprocessing and normalization).

MLOps:
Metaflow: Pipeline step orchestration.
MLflow: Experiment tracking and metric logging (Precision, Recall, F1).
Front-end: HTML5, CSS3 (Flexbox/Transitions), JavaScript & PyScript.
Backend: Uvicorn (ASGI server).

The Pipeline (General Pipeline)
Data Ingestion: Feature extraction from an NPC behavior dataset.
Preprocessing: Manual implementation of One-Hot Encoding and Z-score normalization using JAX.
Modeling: * Linear Classifier: Prediction of base trends.
Logistic Classifier: Action classification using the Sigmoid function.
Evaluation: Metric comparison via MLflow.
