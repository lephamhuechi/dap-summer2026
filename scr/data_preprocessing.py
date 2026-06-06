"""
data_preprocessing.py — E-commerce Dynamic Pricing
Dataset: ecommerce_dataset_updated.csv
Columns: User_ID, Product_ID, Category, Price (Rs.), Discount (%), Final_Price(Rs.), Payment_Method, Purchase_Date

Changes from ride-sharing original:
  - Removed Vehicle_Type / Number_of_Riders / Number_of_Drivers mapping
  - Added log-price and log-final-price features for elasticity regression
  - Added price_bin, discount_bin columns for CR/margin analysis
  - Added purchase volume proxy per (Category, price_bin) for demand modeling
  - Added time features (month, day_of_week) for seasonality controls
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Load and do minimal type coercion on raw CSV."""
    df = pd.read_csv(filepath, parse_dates=["Purchase_Date"], dayfirst=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Returns DataFrame with columns:
        original columns + log_price, log_final_price, margin_pct,
        price_bin, discount_bin, month, day_of_week,
        category_encoded, purchase_count (demand proxy),
        demand_multiplier, supply_multiplier, adjusted_price
    """
    df = df.copy()

    # ── 1. Rename columns for convenience ────────────────────────────────────
    df = df.rename(columns={
        "Price (Rs.)":      "price",
        "Discount (%)":     "discount_pct",
        "Final_Price(Rs.)": "final_price",
    })

    # ── 2. Handle missing values ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    cat_cols = ["Category", "Payment_Method"]
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # ── 3. Outlier capping (IQR) on price columns ─────────────────────────────
    for col in ["price", "final_price"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lb, ub)

    # ── 4. Log-price features (for log-log elasticity regression) ────────────
    df["log_price"]       = np.log(df["price"].clip(lower=0.01))
    df["log_final_price"] = np.log(df["final_price"].clip(lower=0.01))

    # ── 5. Margin proxy: (final_price - assumed_cost) / final_price ──────────
    #    We estimate cost as final_price at 0% discount (i.e. price * 0.55
    #    as a conservative cost ratio). Adjust the ratio to your actual COGS.
    COST_RATIO = 0.45
    df["estimated_cost"] = df["price"] * COST_RATIO
    df["margin_pct"] = ((df["final_price"] - df["estimated_cost"]) /
                        df["final_price"] * 100).round(2)

    # ── 6. Binned features ────────────────────────────────────────────────────
    df["price_bin"] = pd.qcut(df["price"], q=5, labels=[
        "very_low", "low", "mid", "high", "very_high"
    ])
    df["discount_bin"] = pd.cut(df["discount_pct"],
                                bins=[-1, 0, 10, 20, 35, 100],
                                labels=["no_disc", "light", "moderate",
                                        "heavy", "deep"])

    # ── 7. Time features ──────────────────────────────────────────────────────
    if "Purchase_Date" in df.columns:
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], dayfirst=True,
                                             errors="coerce")
        df["month"]        = df["Purchase_Date"].dt.month
        df["day_of_week"]  = df["Purchase_Date"].dt.dayofweek  # 0=Mon
    else:
        df["month"] = 1
        df["day_of_week"] = 0

    # ── 8. Demand proxy: purchase count per (Category, price_bin) ────────────
    #    In ride-sharing the dataset had real-time rider/driver counts.
    #    Here we use within-cell purchase volume as a demand signal.
    demand_counts = (df.groupby(["Category", "price_bin"], observed=True)
                       .transform("count")["price"])
    df["purchase_count"] = demand_counts

    # ── 9. Encode category ────────────────────────────────────────────────────
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["Category"])

    return df
