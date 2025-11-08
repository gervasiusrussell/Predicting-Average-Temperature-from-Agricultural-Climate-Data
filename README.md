# Predicting Average Temperature from Agricultural & Climate Data

**Team:** BOD BCA TEAM  
**Public Kaggle Score:** 82.385  
**Local OOF MAPE:** 84.88

## 📋 Project Overview

This project attempts to predict the average temperature (`Suhu_Rata_Rata_C`) using a tabular dataset of agricultural, economic, and climate features.

The core of this project is a deep dive into feature engineering and hyperparameter tuning (Optuna) for an XGBoost model. A significant finding was that a low local **MAPE (0.8488)** was achieved by a **severely underfitting** model (`max_depth=1`) that "cheated" the metric. The model failed to learn the true patterns, resulting in a poor public Kaggle score (82.385), which is a more accurate reflection of its performance.

## 🧮 Methodology Pipeline

The project is broken down into several key stages:

1.  **Exploratory Data Analysis (EDA):**
    * **Numerical Features:** Analyzed distributions, skewness, and kurtosis. Found most features to be uniformly distributed (low kurtosis), suggesting generalization might be difficult but robust.
    * **Categorical Features:** Analyzed value counts and used Chi-Squared tests to find associations (e.g., `Nama_Negara` and `Wilayah`).
    * **Correlations:** Found very low linear correlation between features and the target, indicating a non-linear model (like XGBoost) is necessary.

2.  **Advanced Feature Engineering:**
    This was the primary focus, executed in several waves:
    * **FE v1/v2 (Golden Features):**
        * **Polynomials:** `Hasil_Panen_sq`, `Emisi_CO2_sq`
        * **Ratios:** `Rasio_Panen_ke_Pupuk`, `Rasio_CO2_ke_Panen`
        * **Interactions:** `Pupuk_x_Pestisida`, `Kesehatan_x_Irigasi`
        * **Deviations:** `Deviasi_Panen_per_Negara` (Difference from the group mean)
    * **FE v3 (Advanced Features):**
        * **Target Encoding:** Encoded categorical features based on their average target value.
        * **Z-Score per Group:** `ZScore_Panen_per_Negara` (A superior version of the Deviation feature, as it accounts for variance).
        * **K-Means Clustering:** Created a `Cluster_ID` feature based on agricultural practice features (e.g., `Penggunaan_Pupuk_KG_per_HA`, `Akses_Irigasi`).
        * **Complex Interactions:** 3-way interactions (`Pupuk_x_Pest_Kesehatan`) and cubic polynomials (`Panen_p3`).
    * **Final Feature Selection:**
        * Combined all engineered features.
        * Programmatically removed redundant or less effective features (e.g., removed `Deviasi_` features in favor of `ZScore_` features, removed original categorical strings).

3.  **Model Training & Tuning (Optuna):**
    * **Model:** XGBoost (`tree_method='hist'`, `enable_categorical=True`).
    * **Validation:** 5-Fold Cross-Validation.
    * **Objective Function:** A custom `pseudo_huber_mape_obj` was used to create a smooth, stable, and differentiable version of MAPE for optimization.
    * **Hyperparameter Tuning:** Used **Optuna** to efficiently search the parameter space.
    * **Leak-Free FE:** To prevent data leakage, **K-Means** and **Z-Score** features were generated *inside* the CV loop. The model was fit only on the fold's training data, then used to transform the validation data.

4.  **Final Model Retraining:**
    * The best parameters from the Optuna study were used to retrain a 5-fold model.
    * **Weighted Training:** To combat the model's failure to predict extreme temperatures, a `WEIGHT_MULTIPLIER = 15` was applied to all data points where the temperature was `< 0°C` or `> 25°C`.

## 📉 Model Evaluation & Critical Analysis

This is the most important finding of the project.

* **The Contradiction:** The final weighted model achieved an OOF (Out-of-Fold) **MAPE of 0.8488**, which appears to be an excellent score. However, the model performed poorly on the public leaderboard (**82.385**).
* **The Root Cause (Underfitting):** The Optuna study found the best-performing model had `max_depth=1`. This is a "decision stump" (a tree with only one split), which is an extremely simple, high-bias model.
* **Visual Diagnosis:** Plotting `Actual` vs. `Predicted` values revealed the model was **severely underfitting**. It only predicted values in three narrow bands (around 0°C, 5°C, and 15°C) and completely failed to predict any high temperatures, despite the weighted training.

### Hypothesis: How the Model "Cheated" MAPE

The low MAPE score is misleading due to the metric's sensitivity.

1.  **MAPE's Weakness:** MAPE explodes (e.g., `(15 - 0.1) / 0.1 = 14900%` error) when the actual value is near zero and the prediction is wrong.
2.  **The "Stupid but Safe" Model:** The `max_depth=1` model learned a simple, "safe" strategy. By predicting near-zero for *all* ambiguous data (both low-temp and high-temp), it avoided these catastrophic MAPE-exploding errors.
3.  **The Trade-off:** It achieved a low MAPE by being **correct on low-value data** (e.g., Actual=0.1, Predicted=0.01) and **horribly wrong, but "percentage-safe," on high-value data** (e.g., Actual=35, Predicted=0.01). The public score of 82.385 reflects this fundamental failure to learn the true patterns.

## 🚀 Next Steps & Future Improvements

The current `max_depth=1` model is a dead end. Future work must focus on forcing a more complex model to learn without "cheating" the metric.

1.  **Rethink Features:** The current features may be insufficient for a deeper tree to separate low-temp and high-temp data.
2.  **Target Transformation:** Train the model on `log1p(Suhu_Rata_Rata_C)` to compress the target range and reduce MAPE's sensitivity near zero. Remember to `expm1()` the final predictions.
3.  **Use a Different Metric:** Use Optuna to optimize for **RMSE** or **MAE** first, as they are less volatile. This will produce a model that is "actually" accurate, even if its MAPE *appears* worse.

## 🛠️ How to Run

1.  **Setup:** Install the required libraries.
    ```bash
    pip install pandas numpy xgboost optuna scikit-learn matplotlib seaborn
    ```
2.  **Data:** Place `train.csv` and `test.csv` in the root directory.
3.  **Execution:** The project is contained within the main Python script/notebook. The script is structured in the following order:
    * EDA
    * Feature Engineering (v2, v3, Combined)
    * Feature Selection
    * Optuna Study (Run `objective(trial)`)
    * Final Model Retraining (Weighted)
    * Model Evaluation
```
