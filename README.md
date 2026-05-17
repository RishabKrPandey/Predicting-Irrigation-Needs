# Precision Agriculture: Predicting Irrigation Needs 🌱

This repository contains a machine learning pipeline designed to predict crop irrigation requirements based on environmental variables and soil metrics. By accurately forecasting irrigation needs, this project aims to optimize water usage and support sustainable precision agriculture.

## 🚀 Project Overview

The core of this project is a robust classification model built to handle tabular environmental data. It leverages gradient-boosted trees and explainable AI techniques to not only make accurate predictions but also provide insights into *why* those predictions are made.

**Key Highlights:**
* **Algorithm:** LightGBM (LGBMClassifier) optimized for tabular data efficiency.
* **Imbalanced Data Handling:** Utilized `class_weight='balanced'` alongside Stratified K-Fold cross-validation to ensure reliable performance across all minority and majority classes.
* **Evaluation:** Achieved an average **Balanced Accuracy of 96.9%** across a 5-fold cross-validation setup.
* **Explainability:** Integrated SHAP (SHapley Additive exPlanations) beeswarm plots to interpret feature importance and model decision-making.

## 🛠️ Tech Stack & Libraries

* **Python 3.12**
* **LightGBM:** Core model training and prediction.
* **Scikit-Learn:** StratifiedKFold cross-validation and pipeline preprocessing.
* **SHAP:** Model interpretability.
* **Pandas & NumPy:** Data manipulation and feature engineering.

## 📊 Methodology

1. **Data Preprocessing:** Standardized and encoded environmental features (e.g., soil moisture, temperature, weather conditions). Categorical text variables were specifically converted to LightGBM's native categorical dtype for faster execution.
2. **Model Training:** Implemented a LightGBM classifier. Gradient boosting was selected as the primary baseline due to its historical superiority in handling complex, non-linear relationships in tabular datasets without requiring extensive feature scaling.
3. **Validation Strategy:** Employed a 5-fold Stratified Cross-Validation approach to prevent overfitting and ensure the model generalizes well to unseen data.

## 🧠 Interpretability & Feature Importance

To crack open the "black box" of the model and understand exactly what drives the predictions, SHAP (SHapley Additive exPlanations) values were calculated. 

<img width="884" height="568" alt="image" src="https://github.com/user-attachments/assets/78f5a807-2b9b-4274-a789-5f2e9a8b493a" />

## Key Insights from SHAP Analysis

- Crop Growth Stage emerged as the most influential factor affecting the model’s predictions, highlighting the importance of stage-specific crop management.
- Soil Moisture showed a strong impact on predictions, indicating that real-time soil water availability is critical for accurate agricultural decision-making.
- Mulching Practices significantly improved model outcomes, suggesting their effectiveness in conserving soil moisture and reducing environmental stress.
- Wind Speed and Temperature were important environmental variables influencing crop conditions and irrigation requirements.
- Rainfall had relatively lower importance compared to soil moisture, implying that retained soil water matters more than total rainfall received.
- Humidity and Previous Irrigation contributed moderately, while Water Source had minimal influence on predictions.
- The model emphasizes field-level conditions and precision agriculture factors over generalized climatic variables.
- SHAP explainability helped identify the most impactful agricultural features, improving model interpretability and trustworthiness.

## 🔮 Future Work

* **Architectural Comparisons:** Experiment with deep learning approaches, specifically evaluating how different neural network architectures perform on this tabular data. I plan to explore how tweaking learnable parameters and adjusting dropout effects might impact the model's variance compared to the current LightGBM baseline.
* **Hyperparameter Tuning:** Implement Optuna or GridSearchCV for exhaustive hyperparameter optimization.
* **Deployment:** Wrap the final model in a lightweight FastAPI application for real-time inference.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/agri-irrigation-model.git](https://github.com/yourusername/agri-irrigation-model.git)
   cd agri-irrigation-model
