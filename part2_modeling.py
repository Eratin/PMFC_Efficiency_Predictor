"""
PEMFC Voltage-Efficiency Prediction — Stacking Ensemble Regression
Author: Google Antigravity
Description: Reusable, leak-free, single-process Machine Learning pipeline
             predicting fuel cell voltage efficiency using XGBoost, SVR,
             Random Forest, and Stacking Ensembles.
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =====================================================================
# 1. Custom Regressor Wrapper for Logistic Regression Meta-Learner
# =====================================================================
class LogisticRegressionAsRegressor(BaseEstimator, RegressorMixin):
    """
    Custom wrapper allowing LogisticRegression (a classifier) to be used
    as a continuous regression meta-learner via target quantile discretization
    and expectation weighting: E[y] = sum(p_k * midpoint_k).
    """
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def __init__(self, n_bins=20, random_state=42):
        self.n_bins = n_bins
        self.random_state = random_state
        self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)

    def fit(self, X, y):
        self.bins_ = np.percentile(y, np.linspace(0, 100, self.n_bins + 1))
        self.bins_[-1] += 1e-5
        self.bins_[0] -= 1e-5
        self.classes_midpoints_ = (self.bins_[:-1] + self.bins_[1:]) / 2

        y_binned = np.digitize(y, self.bins_) - 1
        y_binned = np.clip(y_binned, 0, self.n_bins - 1)
        self.classes_ = np.unique(y_binned)
        self.present_midpoints_ = self.classes_midpoints_[self.classes_]

        self.model.fit(X, y_binned)
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X)
        expected_val = np.sum(proba * self.present_midpoints_, axis=1)
        return expected_val


# =====================================================================
# 2. Main Execution Pipeline
# =====================================================================
def main():
    print("=" * 70)
    print(" PEMFC VOLTAGE-EFFICIENCY PREDICTION PIPELINE")
    print("=" * 70)

    # A. Paths Setup
    data_path = r"d:\DOWNLOAD\b_tech_project\pemfc_efficiency_clean.xlsx"
    output_dir = r"d:\DOWNLOAD\b_tech_project\outputs"
    os.makedirs(output_dir, exist_ok=True)

    # B. Load Dataset
    print("\n[Step 1] Loading Dataset...")
    df = pd.read_excel(data_path, sheet_name="Clean_Data")
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # C. Leak-Free Feature Selection
    features = [
        "current_density_mA_cm2",
        "cathode_pressure_diff_kPa",
        "anode_pressure_diff_kPa",
        "anode_avg_temp_C",
        "cathode_avg_temp_C"
    ]
    target = "voltage_efficiency_pct"

    X = df[features]
    y = df[target]

    # D. Train/Test Split (80/20) & Scaling
    print("\n[Step 2] Preprocessing (80/20 Split & StandardScaler)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    scaler_file = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_file)
    print(f"Saved scaler to {scaler_file}")

    # E. Initialize Base Learners (Tuned Hyperparameters, Single-Process n_jobs=1)
    print("\n[Step 3] Initializing and Fitting Base Learners...")
    best_xgb = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )

    best_svr = SVR(
        kernel="rbf",
        gamma="auto",
        epsilon=0.1,
        C=100
    )

    best_rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=1
    )

    t0 = time.time()
    best_xgb.fit(X_train_scaled, y_train)
    best_svr.fit(X_train_scaled, y_train)
    best_rf.fit(X_train_scaled, y_train)
    print(f"Base learners fitted in {time.time() - t0:.2f}s.")

    # F. Stacking Ensemble Construction
    print("\n[Step 4] Training Stacking Ensemble Regressors...")
    base_estimators = [
        ("xgb", best_xgb),
        ("svr", best_svr),
        ("rf", best_rf)
    ]

    # Stacking with Ridge Meta-Learner (Standard Regression)
    stack_ridge = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(random_state=42),
        cv=5,
        n_jobs=1
    )

    # Stacking with Logistic Regression Meta-Learner (Requested Design)
    stack_logreg = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LogisticRegressionAsRegressor(n_bins=20, random_state=42),
        cv=5,
        n_jobs=1
    )

    print("  Fitting Stacking (Ridge Meta)...")
    stack_ridge.fit(X_train_scaled, y_train)

    print("  Fitting Stacking (LogReg Meta)...")
    stack_logreg.fit(X_train_scaled, y_train)
    print("Stacking ensembles trained successfully.")

    # G. Model Performance Evaluation
    print("\n[Step 5] Evaluating Models on Test Set...")
    models = {
        "XGBoost": best_xgb,
        "SVR": best_svr,
        "Random Forest": best_rf,
        "Stacked (Ridge)": stack_ridge,
        "Stacked (LogReg)": stack_logreg
    }

    metrics_list = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics_list.append({
            "Model": name,
            "R2": r2,
            "RMSE (%)": rmse,
            "MAE (%)": mae,
            "MAPE (%)": mape * 100
        })

    metrics_df = pd.DataFrame(metrics_list).sort_values(by="RMSE (%)").reset_index(drop=True)
    metrics_df["Rank"] = metrics_df.index + 1

    print("\n" + "=" * 75)
    print("TEST SET PERFORMANCE COMPARISON TABLE")
    print("=" * 75)
    print(metrics_df.to_string(index=False))
    print("=" * 75)

    metrics_csv = os.path.join(output_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics table to: {metrics_csv}")

    # H. Visualizations (Saved & Displayed with plt.show())
    print("\n[Step 6] Generating and Displaying Visualizations...")

    # 1. Feature Importance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xgb_fi = best_xgb.feature_importances_
    rf_fi = best_rf.feature_importances_
    svr_pi = permutation_importance(best_svr, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=1)
    svr_fi = svr_pi.importances_mean

    def plot_fi(fi, ax, title, color):
        idx = np.argsort(fi)[::-1]
        ax.bar(range(len(fi)), fi[idx], color=color, edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(fi)))
        ax.set_xticklabels([features[i] for i in idx], rotation=35, ha="right", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Importance Score", fontsize=11)

    plot_fi(xgb_fi, axes[0], "XGBoost Feature Importance", "#1f77b4")
    plot_fi(rf_fi, axes[1], "Random Forest Feature Importance", "#2ca02c")
    plot_fi(svr_fi, axes[2], "SVR Permutation Importance", "#ff7f0e")

    plt.tight_layout()
    fi_file = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_file, dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Residuals & Parity Plots
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 20))
    for i, (name, y_pred) in enumerate(predictions.items()):
        # Parity Plot
        ax1 = axes[i, 0]
        ax1.scatter(y_test, y_pred, alpha=0.30, s=15, color="#1f77b4", edgecolors="none")
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax1.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Ideal y = x")
        ax1.set_title(f"{name}: Predicted vs Actual", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Actual Voltage Efficiency (%)", fontsize=10)
        ax1.set_ylabel("Predicted Voltage Efficiency (%)", fontsize=10)
        ax1.legend(loc="upper left", fontsize=9)

        # Residual Plot
        ax2 = axes[i, 1]
        res = y_test - y_pred
        ax2.scatter(y_pred, res, alpha=0.30, s=15, color="#d62728", edgecolors="none")
        ax2.axhline(0, color="black", linestyle="--", lw=1.5)
        ax2.set_title(f"{name}: Residuals vs Fitted", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Fitted Value (%)", fontsize=10)
        ax2.set_ylabel("Residual (%)", fontsize=10)

    plt.tight_layout()
    res_file = os.path.join(output_dir, "residual_plots.png")
    plt.savefig(res_file, dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Model Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metric_cols = ["R2", "RMSE (%)", "MAE (%)", "MAPE (%)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        bars = ax.bar(metrics_df["Model"], metrics_df[col], color=colors[i], edgecolor="black", alpha=0.85)
        ax.set_title(f"{col} Comparison", fontsize=13, fontweight="bold")
        ax.set_xticklabels(metrics_df["Model"], rotation=25, ha="right", fontsize=10)
        for bar in bars:
            h = bar.get_height()
            offset = 0.01 if col == "R2" else h * 0.02
            ax.text(bar.get_x() + bar.get_width()/2., h + offset,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comp_file, dpi=300, bbox_inches="tight")
    plt.show()

    # I. Save Trained Model Artifacts
    print("\n[Step 7] Serializing and Saving Trained Models...")
    joblib.dump(best_xgb, os.path.join(output_dir, "model_xgboost.joblib"))
    joblib.dump(best_svr, os.path.join(output_dir, "model_svr.joblib"))
    joblib.dump(best_rf, os.path.join(output_dir, "model_rf.joblib"))
    joblib.dump(stack_ridge, os.path.join(output_dir, "model_stack_ridge.joblib"))
    joblib.dump(stack_logreg, os.path.join(output_dir, "model_stack_logreg.joblib"))
    print("All models serialized successfully in:", output_dir)

    print("\n✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
