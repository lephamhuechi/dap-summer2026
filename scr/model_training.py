"""
model_training.py — E-commerce Dynamic Pricing
Dataset: ecommerce_dataset_updated.csv

Changes from ride-sharing original:
  - Target: adjusted_price (computed by dynamic_pricing.py pipeline)
  - Features: log_price, discount_pct, category_encoded, month,
              purchase_count, day_of_week, margin_pct
  - Added feature importance printout (key for pricing team interpretation)
  - Added cross-val R² for more robust evaluation on small dataset (3.6k rows)
  - Model saved to model.pkl by default
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_preprocessing import load_data, preprocess_data
from dynamic_pricing import calculate_dynamic_pricing


# ── Feature config ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_price",          # base price signal (log-transformed)
    "discount_pct",       # current discount level
    "category_encoded",   # category (label-encoded)
    "month",              # seasonality
    "day_of_week",        # weekly seasonality
    "purchase_count",     # demand proxy
    "margin_pct",         # current margin
]

TARGET_COL = "adjusted_price"   # dynamic price computed by pipeline


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    data_path: str = "ecommerce_dataset_updated.csv",
    model_type: str = "random_forest",     # "random_forest" | "gradient_boost"
    save_path: str = "model.pkl",
) -> tuple:
    """
    Load → preprocess → compute dynamic price → train → evaluate → save.

    Returns: (model, feature_names, evaluation_metrics_dict)
    """
    # 1. Load & preprocess
    print("Loading data...")
    df = load_data(data_path)
    df = preprocess_data(df)

    # 2. Compute dynamic pricing (creates adjusted_price column)
    print("Computing dynamic pricing...")
    df, elasticities = calculate_dynamic_pricing(df)

    print("\n── Elasticity estimates by category ──")
    for cat, elast in elasticities.items():
        tag = "elastic" if elast is not None and elast < -1 else "inelastic"
        print(f"  {cat:<18} β = {elast!s:<8}  [{tag}]")

    # 3. Prepare features
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train
    print(f"\nTraining {model_type} on {len(X_train)} samples...")
    if model_type == "gradient_boost":
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
        )

    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred   = model.predict(X_test)
    cv_r2    = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "r2":      round(r2_score(y_test, y_pred), 4),
        "rmse":    round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "mae":     round(mean_absolute_error(y_test, y_pred), 2),
        "cv_r2":   round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
    }

    print("\n── Model evaluation ──")
    print(f"  R²         : {metrics['r2']}")
    print(f"  RMSE       : ₹{metrics['rmse']}")
    print(f"  MAE        : ₹{metrics['mae']}")
    print(f"  CV R² (5-fold): {metrics['cv_r2']} ± {metrics['cv_r2_std']}")

    # 6. Feature importance
    print("\n── Feature importance ──")
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {bar} {imp:.3f}")

    # 7. Save
    joblib.dump({"model": model, "features": FEATURE_COLS}, save_path)
    print(f"\nModel saved → {save_path}")

    return model, FEATURE_COLS, metrics


if __name__ == "__main__":
    train_model(data_path="../data/ecommerce_dataset_updated.csv")
