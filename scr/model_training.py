import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed — skipping XGBRegressor")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠  lightgbm not installed — skipping LGBMRegressor")

from data_preprocessing import load_data, preprocess_data
from dynamic_pricing import calculate_dynamic_pricing

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/ecommerce_dataset_updated.csv"
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "log_price",          # base price signal (log-transformed)
    "discount_pct",       # current discount level
    "category_encoded",   # category (label-encoded)
    "month",              # seasonality
    "day_of_week",        # weekly seasonality
    "purchase_count",     # demand proxy
    "margin_pct",         # current margin
]

FEATURE_LABELS = {
    "log_price":        "Log-price",
    "discount_pct":     "Discount %",
    "category_encoded": "Category",
    "month":            "Month",
    "day_of_week":      "Day of week",
    "purchase_count":   "Purchase count",
    "margin_pct":       "Margin %",
}

TARGET_COL = "adjusted_price"
RANDOM_STATE = 42
CV_FOLDS = 5


# ── Model registry — sklearn Pipeline([scaler, model]) ───────────────────────
# Pipeline đảm bảo scaler chỉ fit trên X_train, tránh data leakage sang X_test

def build_model_registry() -> dict:
    """
    Trả về dict {name: Pipeline} cho tất cả models cần so sánh.
    Mỗi pipeline gồm StandardScaler + estimator.
    Tree-based models (RF, GB, XGB, LGB) không cần scale nhưng
    đưa vào Pipeline cho nhất quán và tránh leakage.
    """
    registry = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.1, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    if HAS_XGB:
        registry["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ])

    if HAS_LGB:
        registry["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )),
        ])

    return registry


# ── Train & evaluate all models ───────────────────────────────────────────────

def evaluate_all_models(
    X_train, X_test, y_train, y_test, X_full, y_full
) -> dict:
    """
    Train từng model trong registry, tính R², RMSE, MAE, CV R².
    Trả về dict {name: metrics_dict}.
    """
    registry = build_model_registry()
    results = {}

    print(f"\n{'─'*62}")
    print(f"  {'Model':<22} {'R²':>6}  {'RMSE':>8}  {'MAE':>8}  {'CV R²':>8}")
    print(f"{'─'*62}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in registry.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        cv   = cross_val_score(pipeline, X_full, y_full,
                               cv=kf, scoring="r2", n_jobs=-1)

        results[name] = {
            "pipeline": pipeline,
            "r2":       round(r2, 4),
            "rmse":     round(rmse, 2),
            "mae":      round(mae, 2),
            "cv_r2":    round(cv.mean(), 4),
            "cv_std":   round(cv.std(), 4),
            "y_pred":   y_pred,
        }

        print(f"  {name:<22} {r2:>6.4f}  ₹{rmse:>7.2f}  ₹{mae:>7.2f}  "
              f"{cv.mean():>6.4f}±{cv.std():.4f}")

    print(f"{'─'*62}")
    return results


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict):
    """
    Chart 9a: grouped bar — R² và CV R² cho từng model.
    """
    names   = list(results.keys())
    r2s     = [v["r2"]    for v in results.values()]
    cv_r2s  = [v["cv_r2"] for v in results.values()]
    rmses   = [v["rmse"]  for v in results.values()]

    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left — R² comparison
    b1 = ax1.bar(x - w/2, r2s,    w, label="Test R²",   color="#3266ad", alpha=0.85, zorder=3)
    b2 = ax1.bar(x + w/2, cv_r2s, w, label=f"CV R² ({CV_FOLDS}-fold)", color="#1D9E75", alpha=0.85, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax1.set_ylabel("R²", fontsize=11)
    ax1.set_title("Model comparison — R²", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_facecolor("#fafaf8")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Right — RMSE comparison
    colors = ["#3266ad" if r == min(rmses) else "#9aa3ad" for r in rmses]
    ax2.bar(x, rmses, color=colors, alpha=0.85, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax2.set_ylabel("RMSE (₹)", fontsize=11)
    ax2.set_title("Model comparison — RMSE (lower = better)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_facecolor("#fafaf8")
    for i, r in enumerate(rmses):
        ax2.text(i, r + max(rmses)*0.01, f"₹{r:.1f}",
                 ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("All Models — Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09a_model_comparison.png", dpi=150)
    plt.close()
    print("✓ Chart 9a: model comparison")


def plot_feature_importance(best_name: str, best_pipeline: Pipeline, metrics: dict):
    """
    Chart 9b: feature importance cho tree-based model tốt nhất.
    Chỉ chạy nếu model có thuộc tính feature_importances_.
    """
    estimator = best_pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        print(f"  (skip Chart 9b — {best_name} has no feature_importances_)")
        return

    importances = sorted(
        zip(FEATURE_COLS, estimator.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    feats  = [FEATURE_LABELS.get(f, f) for f, _ in importances]
    scores = [s for _, s in importances]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(feats))
    colors = ["#3266ad"] + ["#9aa3ad"] * (len(feats) - 1)

    ax.barh(y, scores, color=colors, height=0.6, zorder=3)
    ax.invert_yaxis()

    for i, s in enumerate(scores):
        ax.text(s + max(scores) * 0.015, i, f"{s:.1%}",
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=11)
    ax.set_xlabel("Importance (Gini)", fontsize=11)
    ax.set_title(
        f"Feature Importance — {best_name}\n"
        f"(target: {TARGET_COL} · R²={metrics['r2']:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(scores) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#fafaf8")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09b_feature_importance.png", dpi=150)
    plt.close()
    print("✓ Chart 9b: feature importance")


def plot_actual_vs_predicted(best_name: str, y_test, y_pred, metrics: dict):
    """
    Chart 9c: Actual vs Predicted scatter cho best model.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color="#3266ad", edgecolors="none", zorder=3)

    lo = min(y_test.min(), y_pred.min()) * 0.95
    hi = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit", zorder=4)

    ax.set_xlabel("Actual adjusted_price (₹)", fontsize=11)
    ax.set_ylabel("Predicted adjusted_price (₹)", fontsize=11)
    ax.set_title(
        f"Actual vs Predicted — {best_name}\n"
        f"R²={metrics['r2']:.4f}  RMSE=₹{metrics['rmse']:.2f}  MAE=₹{metrics['mae']:.2f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#fafaf8")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09c_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("✓ Chart 9c: actual vs predicted")


def plot_residuals(best_name: str, y_test, y_pred):
    """
    Chart 9d: Residual plot — phát hiện heteroscedasticity hoặc bias.
    """
    residuals = y_test.values - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_pred, residuals, alpha=0.35, s=18, color="#533AB7",
                edgecolors="none", zorder=3)
    ax1.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Predicted price (₹)", fontsize=11)
    ax1.set_ylabel("Residual (₹)", fontsize=11)
    ax1.set_title(f"Residuals vs Fitted — {best_name}", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#fafaf8")

    ax2.hist(residuals, bins=40, color="#3266ad", alpha=0.8, edgecolor="none")
    ax2.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (₹)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#fafaf8")

    fig.suptitle(f"Residual Analysis — {best_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09d_residuals.png", dpi=150)
    plt.close()
    print("✓ Chart 9d: residual analysis")


# ── Final report ──────────────────────────────────────────────────────────────

def print_training_report(results: dict, best_name: str):
    print("\n" + "═"*62)
    print("  MODEL TRAINING REPORT — Dynamic Pricing (Project 12)")
    print("═"*62)

    print(f"\n── All models ranked by CV R² ──")
    ranked = sorted(results.items(), key=lambda x: x[1]["cv_r2"], reverse=True)
    print(f"  {'Rank':<5} {'Model':<22} {'Test R²':>8} {'RMSE':>9} {'MAE':>9} {'CV R²':>9}")
    print(f"  {'─'*5} {'─'*22} {'─'*8} {'─'*9} {'─'*9} {'─'*9}")
    for rank, (name, m) in enumerate(ranked, 1):
        marker = " ← best" if name == best_name else ""
        print(f"  {rank:<5} {name:<22} {m['r2']:>8.4f} ₹{m['rmse']:>7.2f} ₹{m['mae']:>7.2f} "
              f"{m['cv_r2']:>7.4f}±{m['cv_std']:.4f}{marker}")

    bm = results[best_name]
    print(f"\n── Best model: {best_name} ──")
    print(f"  Test R²       : {bm['r2']}")
    print(f"  RMSE          : ₹{bm['rmse']}")
    print(f"  MAE           : ₹{bm['mae']}")
    print(f"  CV R² (5-fold): {bm['cv_r2']} ± {bm['cv_std']}")
    print(f"  Saved          → model.pkl")

    print("\n── Notes ──")
    print("  • Pipeline([scaler, model]) — scaler fit chỉ trên X_train,")
    print("    không có data leakage từ X_test vào scaling parameters.")
    print("  • Tree-based models (RF, GB, XGB, LGB) không cần scale")
    print("    nhưng vẫn wrap trong Pipeline để interface nhất quán.")
    print("  • Target = adjusted_price (demand×supply multiplier).")
    print("    R² cao là expected vì target được derive từ features.")
    print("    Validate thêm trên held-out time period thực tế.")
    print("  • Retrain quarterly khi có transaction data mới.")
    print("\n" + "═"*62)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_model(
    data_path: str = DATA_PATH,
    save_path: str = "model.pkl",
    best_by: str = "cv_r2",        # "cv_r2" | "r2" | "rmse"
) -> tuple:
    """
    Full pipeline:
      1. Load & preprocess data
      2. Compute dynamic pricing (creates adjusted_price target)
      3. Train 5–7 models via sklearn Pipeline (tránh data leakage)
      4. Compare R², RMSE, MAE, CV R²
      5. Save best model → model.pkl
      6. Generate 4 charts (comparison, feature importance, actual vs pred, residuals)

    Returns: (best_pipeline, feature_names, all_results_dict)
    """
    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(data_path)
    df = preprocess_data(df)

    # ── 2. Compute dynamic pricing target ────────────────────────────────────
    print("Computing dynamic pricing...")
    df, elasticities = calculate_dynamic_pricing(df)

    df.to_csv(
        "data/dynamic_pricing_results.csv",
        index=False
    )

    print("Saved dynamic pricing dataset")

    print("\n── Elasticity estimates by category ──")
    for cat, elast in elasticities.items():
        tag = "elastic  " if elast is not None and elast < -1 else "inelastic"
        print(f"  {cat:<18} β = {str(elast):<8}  [{tag}]")

    # ── 3. Prepare features ──────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size : {len(X_train):,}")
    print(f"Test size  : {len(X_test):,}")
    print(f"Features   : {FEATURE_COLS}")
    print(f"Target     : {TARGET_COL}")

    # ── 4. Train & evaluate all models ──────────────────────────────────────
    print(f"\nTraining {len(build_model_registry())} models...")
    results = evaluate_all_models(X_train, X_test, y_train, y_test, X, y)

    # ── 5. Pick best model ───────────────────────────────────────────────────
    if best_by == "rmse":
        best_name = min(results, key=lambda k: results[k]["rmse"])
    else:
        best_name = max(results, key=lambda k: results[k][best_by])

    best_pipeline = results[best_name]["pipeline"]
    best_metrics  = results[best_name]

    # ── 6. Charts ────────────────────────────────────────────────────────────
    print("\nGenerating charts → charts/")
    plot_model_comparison(results)
    plot_feature_importance(best_name, best_pipeline, best_metrics)
    plot_actual_vs_predicted(
        best_name, y_test, best_metrics["y_pred"], best_metrics
    )
    plot_residuals(best_name, y_test, best_metrics["y_pred"])

    # ── 7. Save best model ───────────────────────────────────────────────────
    joblib.dump({
        "model":        best_pipeline,
        "features":     FEATURE_COLS,
        "best_name":    best_name,
        "metrics":      {k: v for k, v in best_metrics.items() if k != "pipeline"},
        "all_results":  {
            name: {k: v for k, v in m.items() if k not in ("pipeline", "y_pred")}
            for name, m in results.items()
        },
    }, save_path)
    print(f"\nBest model ({best_name}) saved → {save_path}")

    # ── 8. Report ────────────────────────────────────────────────────────────
    print_training_report(results, best_name)

    return best_pipeline, FEATURE_COLS, results


if __name__ == "__main__":
    train_model(data_path="data/ecommerce_dataset_updated.csv")