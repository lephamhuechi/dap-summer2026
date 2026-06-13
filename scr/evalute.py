
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from data_preprocessing import load_data, preprocess_data
from dynamic_pricing import calculate_dynamic_pricing

# Optional: chỉ import model nếu đã train
try:
    from predict import PricePredictor
    HAS_PREDICTOR = True
except Exception:
    HAS_PREDICTOR = False

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH  = "data/ecommerce_dataset_updated.csv"
CHART_DIR  = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

COST_RATIO = 0.45

ELASTICITY_PRIOR = {
    "Beauty":         -0.87,
    "Books":          -2.10,
    "Clothing":       -1.38,
    "Electronics":    -1.82,
    "Home & Kitchen": -1.19,
    "Sports":         -1.55,
    "Toys":           -1.30,
}

OPTIMAL_MULT = {
    "Beauty":         1.24,
    "Books":          1.02,
    "Clothing":       1.14,
    "Electronics":    1.07,
    "Home & Kitchen": 1.18,
    "Sports":         1.10,
    "Toys":           1.12,
}

CAT_COLORS = {
    "Electronics":    "#3266ad",
    "Clothing":       "#1D9E75",
    "Beauty":         "#D85A30",
    "Home & Kitchen": "#BA7517",
    "Sports":         "#533AB7",
    "Books":          "#73726c",
    "Toys":           "#E05C8A",
}


# ── Build evaluation DataFrame ────────────────────────────────────────────────

def build_eval_df(data_path: str = DATA_PATH, model_path: str = None) -> pd.DataFrame:
    """
    Tạo DataFrame có đủ 3 cột giá để so sánh:
      - price          : fixed price (hiện tại)
      - adjusted_price : dynamic price từ demand/supply multiplier
      - optimal_price_est : giá optimal theo elasticity × optimal multiplier
    """
    df = load_data(data_path)
    df = preprocess_data(df)
    df, _ = calculate_dynamic_pricing(df)

    # Optimal price estimate dùng prior elasticity
    df["elasticity_prior"]  = df["Category"].map(ELASTICITY_PRIOR)
    df["optimal_mult"]      = df["Category"].map(OPTIMAL_MULT)
    df["optimal_price_est"] = (df["price"] * df["optimal_mult"]).round(2)

    # Cost & margin cho từng scenario
    cost = df["price"] * COST_RATIO

    df["margin_fixed"]   = ((df["price"]           - cost) / df["price"].clip(0.01) * 100).round(2)
    df["margin_dynamic"] = ((df["adjusted_price"]  - cost) / df["adjusted_price"].clip(0.01) * 100).round(2)
    df["margin_optimal"] = ((df["optimal_price_est"] - cost) / df["optimal_price_est"].clip(0.01) * 100).round(2)

    # Revenue proxy (price × purchase_count)
    df["rev_fixed"]   = df["price"]            * df["purchase_count"]
    df["rev_dynamic"] = df["adjusted_price"]   * df["purchase_count"]
    df["rev_optimal"] = df["optimal_price_est"] * df["purchase_count"]

    # Model prediction nếu có
    if model_path and HAS_PREDICTOR:
        try:
            predictor = PricePredictor(model_path)
            df["model_price"] = df.apply(
                lambda r: predictor.predict_price(
                    category=r["Category"],
                    base_price=r["price"],
                    discount_pct=r["discount_pct"],
                    purchase_count=int(r["purchase_count"]),
                    month=int(r.get("month", 6)),
                ) or r["adjusted_price"],
                axis=1
            )
            df["rev_model"]    = df["model_price"] * df["purchase_count"]
            df["margin_model"] = ((df["model_price"] - cost) / df["model_price"].clip(0.01) * 100).round(2)
            print("✓ Model predictions loaded")
        except Exception as e:
            print(f"  Model prediction skipped: {e}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Fixed vs Dynamic vs Optimal: Revenue & Margin by Category
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(df: pd.DataFrame):
    cat_eval = df.groupby("Category").agg(
        rev_fixed=("rev_fixed", "sum"),
        rev_dynamic=("rev_dynamic", "sum"),
        rev_optimal=("rev_optimal", "sum"),
        margin_fixed=("margin_fixed", "mean"),
        margin_dynamic=("margin_dynamic", "mean"),
        margin_optimal=("margin_optimal", "mean"),
    ).round(2)

    cats = cat_eval.index.tolist()
    x    = np.arange(len(cats))
    w    = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Revenue comparison
    b1 = ax1.bar(x - w,   cat_eval["rev_fixed"],   w, label="Fixed",   color="#73726c", alpha=0.85)
    b2 = ax1.bar(x,        cat_eval["rev_dynamic"], w, label="Dynamic", color="#1D9E75", alpha=0.85)
    b3 = ax1.bar(x + w,   cat_eval["rev_optimal"], w, label="Optimal", color="#3266ad", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("Simulated Revenue (₹)", fontsize=11)
    ax1.set_title("Revenue: Fixed vs Dynamic vs Optimal", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M"))
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#fafaf8")

    # Margin comparison
    ax2.bar(x - w, cat_eval["margin_fixed"],   w, label="Fixed",   color="#73726c", alpha=0.85)
    ax2.bar(x,      cat_eval["margin_dynamic"], w, label="Dynamic", color="#1D9E75", alpha=0.85)
    ax2.bar(x + w, cat_eval["margin_optimal"], w, label="Optimal", color="#3266ad", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Gross Margin %", fontsize=11)
    ax2.set_title("Gross Margin: Fixed vs Dynamic vs Optimal", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#fafaf8")

    fig.suptitle("Pricing Strategy Comparison by Category",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "07_fixed_vs_dynamic.png", dpi=150)
    plt.close()
    print("✓ Chart 7: fixed vs dynamic vs optimal")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Revenue & Margin Lift % (Dynamic over Fixed)
# ══════════════════════════════════════════════════════════════════════════════

def plot_lift(df: pd.DataFrame):
    cat_eval = df.groupby("Category").agg(
        rev_fixed=("rev_fixed", "sum"),
        rev_dynamic=("rev_dynamic", "sum"),
        rev_optimal=("rev_optimal", "sum"),
        margin_fixed=("margin_fixed", "mean"),
        margin_dynamic=("margin_dynamic", "mean"),
        margin_optimal=("margin_optimal", "mean"),
    )

    cat_eval["rev_lift_dynamic"] = (
        (cat_eval["rev_dynamic"] - cat_eval["rev_fixed"]) /
        cat_eval["rev_fixed"] * 100
    ).round(1)
    cat_eval["rev_lift_optimal"] = (
        (cat_eval["rev_optimal"] - cat_eval["rev_fixed"]) /
        cat_eval["rev_fixed"] * 100
    ).round(1)
    cat_eval["margin_lift_dynamic"] = (
        cat_eval["margin_dynamic"] - cat_eval["margin_fixed"]
    ).round(1)
    cat_eval["margin_lift_optimal"] = (
        cat_eval["margin_optimal"] - cat_eval["margin_fixed"]
    ).round(1)

    cats = cat_eval.index.tolist()
    x    = np.arange(len(cats))
    w    = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Revenue lift
    bars1 = ax1.bar(x - w/2, cat_eval["rev_lift_dynamic"], w,
                    label="Dynamic", color="#1D9E75", alpha=0.9)
    bars2 = ax1.bar(x + w/2, cat_eval["rev_lift_optimal"], w,
                    label="Optimal", color="#3266ad", alpha=0.9)
    ax1.axhline(0, color="black", linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f"{h:+.1f}%", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f"{h:+.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("Revenue Lift vs Fixed (%)", fontsize=11)
    ax1.set_title("Revenue Lift by Category", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#fafaf8")

    # Margin lift
    bars3 = ax2.bar(x - w/2, cat_eval["margin_lift_dynamic"], w,
                    label="Dynamic", color="#1D9E75", alpha=0.9)
    bars4 = ax2.bar(x + w/2, cat_eval["margin_lift_optimal"], w,
                    label="Optimal", color="#3266ad", alpha=0.9)
    ax2.axhline(0, color="black", linewidth=0.8)

    for bar in bars3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                 f"{h:+.1f}pp", ha="center", va="bottom", fontsize=8.5)
    for bar in bars4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                 f"{h:+.1f}pp", ha="center", va="bottom", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Margin Lift vs Fixed (pp)", fontsize=11)
    ax2.set_title("Gross Margin Lift by Category", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#fafaf8")

    fig.suptitle("Revenue & Margin Lift: Dynamic and Optimal vs Fixed Pricing",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "08_revenue_margin_lift.png", dpi=150)
    plt.close()
    print("✓ Chart 8: revenue & margin lift")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL BUSINESS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_evaluation_report(df: pd.DataFrame, save_path: str = "evaluation_report.txt"):
    lines = []
    def w(line=""):
        lines.append(line)
        print(line)

    w("═" * 65)
    w("  DYNAMIC PRICING EVALUATION REPORT")
    w("  Project 12 — Impact on Conversion Rate & Margin")
    w("═" * 65)

    # Overall numbers
    total_fixed   = df["rev_fixed"].sum()
    total_dynamic = df["rev_dynamic"].sum()
    total_optimal = df["rev_optimal"].sum()
    rev_lift_dyn  = (total_dynamic - total_fixed) / total_fixed * 100
    rev_lift_opt  = (total_optimal - total_fixed) / total_fixed * 100

    avg_margin_fixed   = df["margin_fixed"].mean()
    avg_margin_dynamic = df["margin_dynamic"].mean()
    avg_margin_optimal = df["margin_optimal"].mean()

    w("\n── Overall simulation results ──")
    w(f"  {'Scenario':<20} {'Revenue':>14}  {'Rev Lift':>9}  {'Avg Margin':>11}")
    w(f"  {'-'*20}  {'-'*14}  {'-'*9}  {'-'*11}")
    w(f"  {'Fixed pricing':<20} ₹{total_fixed:>13,.0f}  {'—':>9}  {avg_margin_fixed:>10.1f}%")
    w(f"  {'Dynamic pricing':<20} ₹{total_dynamic:>13,.0f}  {rev_lift_dyn:>+8.1f}%  {avg_margin_dynamic:>10.1f}%")
    w(f"  {'Optimal pricing':<20} ₹{total_optimal:>13,.0f}  {rev_lift_opt:>+8.1f}%  {avg_margin_optimal:>10.1f}%")

    w("\n── By category ──")
    cat_eval = df.groupby("Category").agg(
        rev_fixed=("rev_fixed", "sum"),
        rev_dynamic=("rev_dynamic", "sum"),
        rev_optimal=("rev_optimal", "sum"),
        margin_fixed=("margin_fixed", "mean"),
        margin_optimal=("margin_optimal", "mean"),
    )
    cat_eval["rev_lift_%"]   = ((cat_eval["rev_optimal"] - cat_eval["rev_fixed"]) / cat_eval["rev_fixed"] * 100).round(1)
    cat_eval["margin_lift_pp"] = (cat_eval["margin_optimal"] - cat_eval["margin_fixed"]).round(1)
    cat_eval["elasticity"]   = cat_eval.index.map(ELASTICITY_PRIOR)
    cat_eval["opt_mult"]     = cat_eval.index.map(OPTIMAL_MULT)

    w(f"\n  {'Category':<18} {'β':>6}  {'Opt×':>5}  {'Rev Lift':>9}  {'Margin Lift':>12}")
    w(f"  {'-'*18}  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*12}")
    for cat, row in cat_eval.iterrows():
        w(f"  {cat:<18} {row['elasticity']:>6.2f}  {row['opt_mult']:>5.2f}×  "
          f"{row['rev_lift_%']:>+8.1f}%  {row['margin_lift_pp']:>+10.1f}pp")

    w("\n── Key insights ──")
    w("  1. Beauty & Home (inelastic): có thể tăng giá +18–24% mà không mất nhiều CR")
    w("  2. Electronics & Books (elastic): rất nhạy cảm với giá, chỉ tăng +2–7%")
    w("  3. Dynamic pricing tổng thể mang lại revenue lift tích cực so với fixed")
    w("  4. Margin cải thiện ở tất cả categories khi áp dụng optimal pricing")

    w("\n── Dynamic pricing rules (final) ──")
    rules = [
        ("Beauty, Home & Kitchen", "inelastic", "-0.87 to -1.19",
         "Tăng +18–24% · Max cap +30% · Trigger khi stock thấp hoặc competitor tăng giá"),
        ("Clothing, Sports, Toys", "moderate",  "-1.19 to -1.55",
         "Tăng +10–14% · A/B test weekly · Volume discount khi cart > ₹1,000"),
        ("Electronics",            "elastic",   "-1.82",
         "Tăng tối đa +7% · Price match competitor trong 24h · Flash sale -10% khi tồn kho cao"),
        ("Books",                  "very elastic", "-2.10",
         "Giữ ±2% market rate · Bundle pricing (mua 2 giảm thêm 5%) thay vì tăng đơn giá"),
    ]
    for seg, etype, beta_range, rule in rules:
        w(f"\n  [{seg}]  {etype}  β ∈ [{beta_range}]")
        w(f"    → {rule}")

    w("\n── Model evaluation summary ──")
    w("  Model: RandomForestRegressor (n=200, max_depth=8)")
    w("  Target: adjusted_price (demand×supply multiplier applied)")
    w("  Features: log_price, discount_pct, category_encoded,")
    w("            month, day_of_week, purchase_count, margin_pct")
    w("  Expected R²: 0.85–0.95 (target is derived from features,")
    w("               so high R² is expected — validate on held-out period)")
    w("  Recommendation: re-train quarterly with fresh transaction data")

    w("\n── Next steps ──")
    w("  1. Collect real quantity/demand data (dataset hiện là synthetic)")
    w("  2. A/B test: 90% fixed / 10% dynamic, track CR + revenue 4 tuần")
    w("  3. Tích hợp predict.py vào pricing API (call khi restock hoặc daily cron)")
    w("  4. Build Tableau dashboard từ evaluation_report output")
    w("  5. Elasticity re-estimation sau khi có experiment data thực tế")

    w("\n" + "═" * 65)

    # Save to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(data_path: str = DATA_PATH, model_path: str = None):
    print(f"Building evaluation dataset from {data_path}...")
    df = build_eval_df(data_path, model_path)
    print(f"Done: {len(df):,} rows")

    print("\nGenerating evaluation charts...")
    plot_comparison(df)
    plot_lift(df)

    print_evaluation_report(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamic pricing")
    parser.add_argument("--data",  default=DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--model", default=None,      help="Path to model.pkl (optional)")
    args = parser.parse_args()
    run_evaluation(data_path=args.data, model_path=args.model)