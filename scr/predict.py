"""
predict.py — E-commerce Dynamic Pricing
Dataset: ecommerce_dataset_updated.csv

Changes from ride-sharing original:
  - Inputs: category, base_price, discount_pct, purchase_count, month, day_of_week
  - Removed Vehicle_Type, Number_of_Riders, Number_of_Drivers
  - Added rule-based fallback when model isn't available
  - Added predict_batch() for DataFrame input (useful for pricing jobs)
  - Added pricing_recommendation() for human-readable output
"""

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import preprocess_data
from dynamic_pricing import calculate_dynamic_pricing

# Category → label encoding (must match training order from LabelEncoder)
CATEGORY_MAP = {
    "beauty":         0,
    "books":          1,
    "clothing":       2,
    "electronics":    3,
    "home & kitchen": 4,
    "sports":         5,
    "toys":           6,
}

ELASTICITY_PRIORS = {
    "beauty":         -0.87,
    "books":          -2.10,
    "clothing":       -1.38,
    "electronics":    -1.82,
    "home & kitchen": -1.19,
    "sports":         -1.55,
    "toys":           -1.30,
}

OPTIMAL_MULT = {          # estimated optimal price multiplier per category
    "beauty":         1.24,
    "books":          1.02,
    "clothing":       1.14,
    "electronics":    1.07,
    "home & kitchen": 1.18,
    "sports":         1.10,
    "toys":           1.12,
}


class PricePredictor:
    """
    E-commerce price predictor.

    Args:
        model_path: path to saved model.pkl (from model_training.py).
                    If None, uses rule-based fallback.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.features = None

        if model_path:
            try:
                payload       = joblib.load(model_path)
                self.model    = payload["model"]
                self.features = payload["features"]
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Model load failed ({e}) — using rule-based fallback.")
        else:
            print("No model path provided — using rule-based fallback.")

    # ── Single-item prediction ────────────────────────────────────────────────

    def predict_price(
        self,
        category: str,
        base_price: float,
        discount_pct: float = 0.0,
        purchase_count: int = 100,
        month: int = 6,
        day_of_week: int = 2,
        cost_ratio: float = 0.45,
    ) -> float | None:
        """
        Predict optimal dynamic price for one product.

        Args:
            category:       one of CATEGORY_MAP keys (case-insensitive)
            base_price:     original listed price (Rs.)
            discount_pct:   current discount % (0–50)
            purchase_count: demand proxy — purchases in this price/category cell
            month:          1–12
            day_of_week:    0 (Mon) – 6 (Sun)
            cost_ratio:     estimated COGS / base_price

        Returns:
            predicted price (float) or None on error
        """
        try:
            self._validate(category, base_price, discount_pct, purchase_count)

            cat_key = category.lower().strip()
            final_price = base_price * (1 - discount_pct / 100)
            log_price   = np.log(max(final_price, 0.01))
            cat_enc     = CATEGORY_MAP.get(cat_key, 0)
            margin_pct  = ((final_price - base_price * cost_ratio) /
                           max(final_price, 0.01) * 100)

            if self.model is not None:
                X = np.array([[log_price, discount_pct, cat_enc, month,
                                day_of_week, purchase_count, margin_pct]])
                return round(float(self.model.predict(X)[0]), 2)
            else:
                return self._rule_based(cat_key, base_price, purchase_count)

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict prices for a preprocessed DataFrame (output of preprocess_data).
        Adds column 'predicted_price'.
        """
        from data_preprocessing import preprocess_data
        from dynamic_pricing import calculate_dynamic_pricing

        df_proc = preprocess_data(df)
        df_proc, _ = calculate_dynamic_pricing(df_proc)

        if self.model is not None and self.features is not None:
            X = df_proc[self.features].fillna(0)
            df_proc["predicted_price"] = self.model.predict(X).round(2)
        else:
            df_proc["predicted_price"] = df_proc["adjusted_price"]

        return df_proc

    # ── Human-readable recommendation ────────────────────────────────────────

    def pricing_recommendation(
        self,
        category: str,
        base_price: float,
        discount_pct: float = 0.0,
        purchase_count: int = 100,
        month: int = 6,
        cost_ratio: float = 0.45,
    ) -> dict:
        """
        Return a dict with predicted price + context for the pricing team.
        """
        cat_key  = category.lower().strip()
        pred     = self.predict_price(category, base_price, discount_pct,
                                       purchase_count, month)
        elast    = ELASTICITY_PRIORS.get(cat_key, -1.3)
        opt_mult = OPTIMAL_MULT.get(cat_key, 1.10)
        opt_px   = round(base_price * opt_mult, 2)
        cost     = base_price * cost_ratio
        margin   = round(((pred or base_price) - cost) / max(pred or base_price, 0.01) * 100, 1)
        demand_type = "inelastic" if abs(elast) < 1 else "elastic"

        return {
            "category":          category,
            "base_price":        base_price,
            "predicted_price":   pred,
            "optimal_price_est": opt_px,
            "price_elasticity":  elast,
            "demand_type":       demand_type,
            "gross_margin_pct":  margin,
            "recommendation": (
                f"Raise price to ₹{opt_px} (+{round((opt_mult-1)*100)}%). "
                f"Demand is {demand_type} (β={elast}). "
                f"Expected margin: {margin}%."
            ) if pred and pred < opt_px else (
                f"Price is near/above optimal (₹{opt_px}). "
                f"Monitor CR — elasticity β={elast}."
            )
        }

    # ── Validation & fallback ─────────────────────────────────────────────────

    @staticmethod
    def _validate(category, base_price, discount_pct, purchase_count):
        if category.lower().strip() not in CATEGORY_MAP:
            raise ValueError(f"Unknown category '{category}'. "
                             f"Valid: {list(CATEGORY_MAP.keys())}")
        if not (0 < base_price <= 10_000):
            raise ValueError("base_price must be > 0 and ≤ 10,000")
        if not (0 <= discount_pct <= 90):
            raise ValueError("discount_pct must be 0–90")
        if purchase_count < 0:
            raise ValueError("purchase_count must be ≥ 0")

    @staticmethod
    def _rule_based(cat_key: str, base_price: float, purchase_count: int) -> float:
        """Fallback: multiply base price by category-specific optimal multiplier."""
        mult = OPTIMAL_MULT.get(cat_key, 1.10)
        # Boost multiplier slightly if demand is high
        if purchase_count > 120:
            mult = min(mult * 1.05, 1.40)
        return round(base_price * mult, 2)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = PricePredictor()   # rule-based (no saved model yet)

    examples = [
        ("Electronics", 299.99, 5,  80),
        ("Beauty",      89.50,  10, 140),
        ("Books",       45.00,  20, 95),
        ("Clothing",    199.00, 0,  110),
    ]

    print("\n── Pricing recommendations ──")
    for cat, price, disc, vol in examples:
        rec = predictor.pricing_recommendation(cat, price, disc, vol)
        print(f"\n{rec['category']} | base ₹{price}")
        print(f"  Predicted:  ₹{rec['predicted_price']}")
        print(f"  Optimal est: ₹{rec['optimal_price_est']}  (β={rec['price_elasticity']})")
        print(f"  → {rec['recommendation']}")
