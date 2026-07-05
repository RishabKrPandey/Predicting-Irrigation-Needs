# Precision Agriculture: Predicting Irrigation Needs 🌱

This repository contains a machine learning pipeline designed to predict crop irrigation requirements based on environmental variables and soil metrics. By accurately forecasting irrigation needs, this project aims to optimize water usage and support sustainable precision agriculture.

## 🚀 Project Overview

The core of this project is a robust classification model built to handle tabular environmental data. It leverages LightGBM and explainable AI techniques to not only make accurate predictions but also provide insights into *why* those predictions are made.

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

## 🧠 Model Evaluation & Interpretability

To prove the model's reliability and understand its underlying logic, we evaluated both its performance distribution and its feature decision hierarchy.

### 1. Performance Robustness (Confusion Matrix)

<img width="549" height="455" alt="image" src="https://github.com/user-attachments/assets/0a9c528c-ecbf-4b73-9c07-c126bab2c979" />

The confusion matrix demonstrates the model's robustness across all three irrigation classes (Low, Medium, High). 
* **Handling Imbalance:** Thanks to the `class_weight='balanced'` parameter, the model performs exceptionally well across the board, proving it doesn't just default to the majority class.
* **Minimizing Critical Errors:** The matrix shows a strong diagonal (True Positives) and extremely low values in the far corners. This means the model rarely makes catastrophic agricultural mistakes—like predicting a "Low" water need when the crops actually have a "High" need.

### 2. Decision Logic (SHAP Feature Importance)

<img width="884" height="568" alt="image" src="https://github.com/user-attachments/assets/78f5a807-2b9b-4274-a789-5f2e9a8b493a" />

While the confusion matrix proves the model works, the SHAP (SHapley Additive exPlanations) analysis proves *why* it works. The top drivers for the model are:
1. **Crop Growth Stage (+2.73):** The biological phase of the crop dictates baseline water needs.
2. **Soil Moisture (+2.34):** The current water retention in the ground.
3. **Mulching Used (+1.29):** A physical tactic that drastically reduces evaporation.

**Model Trust:** By prioritizing biological metrics and soil retention over basic daily weather data (like Temperature or Wind Speed), the LightGBM classifier proves it has learned genuine, real-world agronomic principles rather than memorizing dataset noise.

## 🗄️ Dataset

The environmental and soil metric data used to train this model was sourced from Kaggle. You can explore the original dataset, including full feature descriptions and data distributions, here: 

🔗 **[Kaggle Dataset](https://www.kaggle.com/competitions/playground-series-s6e4/overview)**

*(Note: To run this notebook locally, download the dataset from the Kaggle link above and place the `.csv` file inside a `data/` folder at the root of this repository.)*

## 🔮 Future Work

* **Architectural Comparisons:** Experiment with deep learning approaches, specifically evaluating how different neural network architectures perform on this tabular data. I plan to explore how tweaking learnable parameters and adjusting dropout effects might impact the model's variance compared to the current LightGBM baseline.
* **Hyperparameter Tuning:** Implement Optuna or GridSearchCV for exhaustive hyperparameter optimization.
* **Deployment:** Wrap the final model in a lightweight FastAPI application for real-time inference.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/agri-irrigation-model.git](https://github.com/yourusername/agri-irrigation-model.git)
   cd agri-irrigation-model
