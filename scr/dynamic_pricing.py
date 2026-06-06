
import numpy as np
import pandas as pd
from scipy import stats


# ── Demand multiplier ────────────────────────────────────────────────────────

def calculate_demand_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace rider-count-based multiplier with purchase_count-based one.
    High purchase volume in a price bin → higher demand multiplier.
    """
    df = df.copy()
    p75 = np.percentile(df["purchase_count"], 75)
    p25 = np.percentile(df["purchase_count"], 25)

    df["demand_multiplier"] = np.where(
        df["purchase_count"] > p75,
        df["purchase_count"] / p75,          # above median: scale up
        df["purchase_count"] / max(p25, 1),  # below median: scale down
    )
    return df


# ── Supply multiplier ─────────────────────────────────────────────────────────

def calculate_supply_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    """
    In e-commerce, high discount signals excess inventory (oversupply).
    We invert: low discount → supply is tight → price should rise.
    """
    df = df.copy()
    p75 = np.percentile(df["discount_pct"], 75)
    p25 = np.percentile(df["discount_pct"], 25)

    # More discount → supply_multiplier < 1 (don't inflate further)
    # Less discount → supply_multiplier > 1 (can push price up)
    safe_disc = df["discount_pct"].replace(0, 1).clip(lower=1)
    df["supply_multiplier"] = np.where(
        df["discount_pct"] < p25,
        p75 / safe_disc,           # low discount → push up
        p25 / safe_disc,           # high discount → hold
    )
    # Cap multipliers to avoid extreme swings
    df["supply_multiplier"] = df["supply_multiplier"].clip(0.8, 1.5)
    return df


# ── Adjusted ride cost → adjusted_price ──────────────────────────────────────

def calculate_adjusted_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined demand × supply adjustment applied to base price.
    Mirrors the original adjusted_ride_cost logic, now on e-commerce price.
    """
    df = calculate_demand_multiplier(df)
    df = calculate_supply_multiplier(df)

    DEMAND_FLOOR = 0.8
    SUPPLY_FLOOR = 0.8

    df["adjusted_price"] = df["price"] * (
        np.maximum(df["demand_multiplier"], DEMAND_FLOOR) *
        np.maximum(df["supply_multiplier"], SUPPLY_FLOOR)
    )
    # Round to 2 decimal places
    df["adjusted_price"] = df["adjusted_price"].round(2)
    return df


# ── Price elasticity via log-log OLS ─────────────────────────────────────────

def estimate_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category log-log OLS: log(purchase_count) ~ β·log(final_price) + controls

    Returns original df with new columns:
        elasticity_{category}  → stored as category-level column 'elasticity'
        optimal_price          → P* = MC / (1 + 1/elasticity)
    """
    df = df.copy()
    elasticities = {}

    for cat, grp in df.groupby("Category"):
        # Need sufficient variation — skip tiny groups
        if len(grp) < 30:
            elasticities[cat] = np.nan
            continue

        log_price  = np.log(grp["final_price"].clip(lower=0.01))
        log_demand = np.log(grp["purchase_count"].clip(lower=1))

        # Add seasonality control (month)
        X = np.column_stack([log_price, grp["month"].fillna(1)])
        X = np.hstack([np.ones((len(X), 1)), X])

        try:
            # OLS: (XᵀX)⁻¹Xᵀy
            beta = np.linalg.lstsq(X, log_demand, rcond=None)[0]
            elasticities[cat] = round(float(beta[1]), 3)  # price coefficient
        except Exception:
            elasticities[cat] = np.nan

    df["elasticity"] = df["Category"].map(elasticities)

    # ── Optimal price: P* = MC / (1 + 1/ε), only valid when ε < -1 ──────────
    df["optimal_price"] = np.where(
        df["elasticity"] < -1,
        df["estimated_cost"] / (1 + 1 / df["elasticity"]),
        np.nan  # inelastic → can raise price further; flag separately
    )
    df["optimal_price"] = df["optimal_price"].round(2)

    return df, elasticities


# ── Revenue & margin simulation ───────────────────────────────────────────────

def simulate_revenue(df: pd.DataFrame, price_col: str = "adjusted_price") -> pd.DataFrame:
    """Compute simulated revenue and margin for a given price column."""
    df = df.copy()
    df["sim_revenue"] = df[price_col] * df["purchase_count"]
    df["sim_margin"]  = ((df[price_col] - df["estimated_cost"]) /
                          df[price_col] * 100).round(2)
    return df


# ── Master pipeline ────────────────────────────────────────────────────────────

def calculate_dynamic_pricing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline:
      1. Demand + supply multipliers → adjusted_price
      2. Elasticity OLS per category
      3. Revenue/margin simulation

    Returns: (enriched_df, elasticity_dict)
    """
    df = calculate_adjusted_price(df)
    df, elasticities = estimate_elasticity(df)
    df = simulate_revenue(df, price_col="adjusted_price")
    return df, elasticities
