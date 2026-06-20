
import warnings
warnings.filterwarnings("ignore")

# Fix path — đảm bảo Python tìm thấy các file cùng thư mục
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from data_preprocessing import load_data, preprocess_data
from dynamic_pricing import calculate_dynamic_pricing

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH  = "data/ecommerce_dataset_updated.csv"
CHART_DIR  = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

COST_RATIO = 0.45

CAT_COLORS = {
    "Electronics":    "#3266ad",
    "Clothing":       "#1D9E75",
    "Beauty":         "#D85A30",
    "Home & Kitchen": "#BA7517",
    "Sports":         "#533AB7",
    "Books":          "#73726c",
    "Toys":           "#E05C8A",
}

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


# ── Load data ─────────────────────────────────────────────────────────────────

def load_pipeline(data_path: str = DATA_PATH):
    df = load_data(data_path)
    df = preprocess_data(df)
    df, elasticities = calculate_dynamic_pricing(df)
    df["elasticity_prior"]  = df["Category"].map(ELASTICITY_PRIOR)
    df["optimal_mult"]      = df["Category"].map(OPTIMAL_MULT)
    df["optimal_price_est"] = (df["price"] * df["optimal_mult"]).round(2)
    return df, elasticities


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Price elasticity by category (horizontal bar + CI)
# ══════════════════════════════════════════════════════════════════════════════

def plot_elasticity(df):
    cats   = list(ELASTICITY_PRIOR.keys())
    betas  = [ELASTICITY_PRIOR[c] for c in cats]
    ci_lo  = [-2.1, -2.5, -1.6, -2.2, -1.4, -1.9, -1.6]
    ci_hi  = [-0.7, -1.7, -1.1, -1.5, -1.0, -1.2, -1.0]
    colors = [CAT_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(cats))

    ax.barh(y, betas, color=colors, height=0.5, zorder=3)

    for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], color="black", linewidth=1.5, zorder=4)
        ax.plot([lo, lo], [i - 0.15, i + 0.15], color="black", linewidth=1.5)
        ax.plot([hi, hi], [i - 0.15, i + 0.15], color="black", linewidth=1.5)

    ax.axvline(-1, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="|ε| = 1  (elastic threshold)")

    for i, b in enumerate(betas):
        ax.text(b - 0.05, i, f"β = {b}", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=11)
    ax.set_xlabel("Price Elasticity (β)", fontsize=11)
    ax.set_title("Price Elasticity by Category\n(Log-Log OLS · 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-2.8, 0.1)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#fafaf8")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_elasticity_by_category.png", dpi=150)
    plt.close()
    print("✓ Chart 1: elasticity by category")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Conversion Rate vs Price Index (CR proxy)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cr_vs_price(df):
    """
    Simulated CR vs Price curve based on elasticity priors (Table 2).

    CR(P) = CR0 * (P / P0) ** beta

    For each category, CR is normalized to 100 at the current average
    price (P0). The curve illustrates the theoretical demand response
    implied by the estimated elasticity coefficients, since the raw
    Kaggle dataset is synthetic and does not contain a real demand/CR
    signal that varies with price (see Discussion, dataset limitations).
    """
    avg_price = df.groupby("Category")["price"].mean()
    avg_margin = df.groupby("Category")["margin_pct"].mean()

    price_mult = np.linspace(0.7, 1.5, 60)  # -30% to +50% of current price

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax2 = ax1.twinx()

    for cat, beta in ELASTICITY_PRIOR.items():
        cr_curve = 100 * (price_mult ** beta)
        ax1.plot(price_mult, cr_curve, color=CAT_COLORS[cat],
                 linewidth=2, label=f"{cat} (β={beta})")

    # Reference vertical line at current price (multiplier = 1.0)
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax1.text(1.01, 102, "current price", fontsize=8, color="black", alpha=0.7)

    # Average margin trend (illustrative, increases with price)
    margin_curve = avg_margin.mean() + (price_mult - 1) * 100 * 0.45
    ax2.plot(price_mult, margin_curve, color="#1D9E75", linewidth=2,
             linestyle=":", label="Avg margin trend")

    ax1.set_xlabel("Price Multiplier (relative to current price)", fontsize=11)
    ax1.set_ylabel("CR Index (100 = current price)", color="#3266ad", fontsize=11)
    ax2.set_ylabel("Gross Margin %", color="#1D9E75", fontsize=11)
    ax1.set_title("Simulated CR vs. Price Curves by Category\n"
                   "(based on elasticity priors, Table 2)",
                   fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#fafaf8")
    ax1.set_xlim(0.7, 1.5)
    ax1.set_ylim(0, max(160, 100 * (0.7 ** min(ELASTICITY_PRIOR.values()))))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_cr_vs_price_bin.png", dpi=150)
    plt.close()
    print("✓ Chart 2: CR vs price (elasticity-based simulation)")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Revenue Simulation Parabola (per category)
# ══════════════════════════════════════════════════════════════════════════════

def plot_revenue_parabola(df):
    """
    Profit simulation curve by category.

    Profit(m) = (P0*m - c) * (v0 * m^beta),  c = COST_RATIO * P0

    Unlike revenue (which is monotonic for constant elasticity), profit
    is a true concave curve (parabola-like) with an interior maximum.
    The maximizing multiplier m* matches the Lerner-rule optimal price
    P* = c / (1 + 1/beta)  =>  m* = P*/P0, marked in red.
    """
    avg_price = df.groupby("Category")["price"].mean()
    avg_vol   = df.groupby("Category")["purchase_count"].mean()
    mults     = np.linspace(0.5, 4.5, 300)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, cat in enumerate(ELASTICITY_PRIOR.keys()):
        ax    = axes[i]
        beta  = ELASTICITY_PRIOR[cat]
        p0    = avg_price.get(cat, 200)
        v0    = avg_vol.get(cat, 100)
        c     = COST_RATIO * p0

        # Profit(m) = (p0*m - c) * v0 * m^beta
        profits = (p0 * mults - c) * v0 * (mults ** beta)

        ax.plot(mults, profits, color=CAT_COLORS[cat], linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)

        if beta < -1:
            # Interior maximum exists (Lerner rule): P* = c / (1 + 1/beta)
            p_star = c / (1 + 1 / beta)
            m_star = p_star / p0

            if 0.5 <= m_star <= 4.5:
                profit_star = (p0 * m_star - c) * v0 * (m_star ** beta)
                ax.axvline(m_star, color="red", linestyle="--", linewidth=1, alpha=0.8)
                ax.scatter([m_star], [profit_star], color="red", zorder=5, s=60)
                ax.annotate(f"m*={m_star:.2f}\u00d7", xy=(m_star, profit_star),
                            xytext=(m_star + 0.06, profit_star),
                            fontsize=8, color="red")
            else:
                # Mathematical optimum lies outside the realistic pricing
                # range; profit is still increasing/decreasing monotonically
                # within [0.5, 2.0].
                ax.text(0.97, 0.05,
                        f"m*={m_star:.2f}\u00d7 (outside range)\n"
                        f"|\u03b2|={abs(beta):.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="#185FA5",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#E6F1FB", ec="none"))
        else:
            # |beta| < 1: inelastic demand -> profit increases monotonically
            # over this range; no interior maximum (demand too unresponsive).
            ax.text(0.97, 0.05,
                    f"|\u03b2|={abs(beta):.2f} < 1\nno interior max\n(inelastic)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#993C1D",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FAECE7", ec="none"))

        ax.set_title(cat, fontsize=10, fontweight="bold", color=CAT_COLORS[cat])
        ax.set_xlabel("Price mult", fontsize=8)
        ax.set_ylabel("Sim. Profit", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"₹{v/1000:.0f}k"))
        ax.set_xlim(0.5, 4.5)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#fafaf8")

    axes[-1].set_visible(False)
    fig.suptitle("Profit Simulation Curve by Category\n"
                  "(red = profit-maximizing price where it exists, Lerner rule)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_revenue_parabola.png", dpi=150)
    plt.close()
    print("✓ Chart 3: revenue parabola")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Gross Margin by Price Bin
# ══════════════════════════════════════════════════════════════════════════════

def plot_margin_by_bin(df):
    margin_data = (df.groupby(["price_bin", "Category"], observed=True)["margin_pct"]
                     .mean().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    margin_data.plot(kind="bar", ax=ax,
                     color=[CAT_COLORS.get(c, "#999") for c in margin_data.columns],
                     width=0.75, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Price Bin", fontsize=11)
    ax.set_ylabel("Gross Margin %", fontsize=11)
    ax.set_title("Gross Margin % by Price Bin & Category",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(margin_data.index.astype(str), rotation=30, ha="right")
    ax.legend(title="Category", fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#fafaf8")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_margin_by_price_bin.png", dpi=150)
    plt.close()
    print("✓ Chart 4: margin by price bin")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — CR × Price Heat Map
# ══════════════════════════════════════════════════════════════════════════════

def plot_cr_heatmap(df):
    heatmap_data = (df.groupby(["Category", "price_bin"], observed=True)
                      .agg(cr_proxy=("purchase_count", "mean"))
                      .reset_index()
                      .pivot(index="Category", columns="price_bin",
                             values="cr_proxy"))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Avg purchase count"})
    ax.set_title("CR Proxy Heat Map — Purchase Count by Category × Price Bin",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Price Bin", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_cr_price_heatmap.png", dpi=150)
    plt.close()
    print("✓ Chart 5: CR-price heat map")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Fixed vs Dynamic Price Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col, label) in zip(axes, [
        ("price",          "Fixed Price (Rs.)"),
        ("adjusted_price", "Dynamic Price (Rs.)"),
    ]):
        for cat in df["Category"].unique():
            subset = df[df["Category"] == cat][col]
            ax.hist(subset, bins=30, alpha=0.55, label=cat,
                    color=CAT_COLORS.get(cat, "#999"), edgecolor="none")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Price (Rs.)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#fafaf8")

    fig.suptitle("Price Distribution: Fixed vs Dynamic Pricing",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_price_distribution.png", dpi=150)
    plt.close()
    print("✓ Chart 6: price distribution fixed vs dynamic")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════
def run_sql_analysis(df, sql_path: str = "sql/queries.sql"):
    import duckdb
    with open(sql_path, encoding="utf-8") as f:
        raw = f.read()

    queries = [q.strip() for q in raw.split(";") if q.strip() and not q.strip().startswith("--")]
    for i, q in enumerate(queries, 1):
        print(f"\n=== SQL Query {i} ===")
        print(duckdb.sql(q).df())

def print_report(df, elasticities):
    print("\n" + "═" * 60)
    print("  DYNAMIC PRICING ANALYSIS REPORT")
    print("═" * 60)

    print("\n── Dataset overview ──")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Categories         : {df['Category'].nunique()}")
    print(f"  Price range        : ₹{df['price'].min():.0f} – ₹{df['price'].max():.0f}")
    print(f"  Avg discount       : {df['discount_pct'].mean():.1f}%")

    print("\n── Elasticity (log-log OLS prior) ──")
    print(f"  {'Category':<18} {'β':>6}  {'Type':<12}  {'Opt mult':>8}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*12}  {'-'*8}")
    for cat, beta in ELASTICITY_PRIOR.items():
        tag = "inelastic" if abs(beta) < 1 else "elastic  "
        opt = OPTIMAL_MULT[cat]
        print(f"  {cat:<18} {beta:>6.2f}  {tag}  {opt:>8.2f}×")

    print("\n── Price adjustment summary ──")
    cat_summary = df.groupby("Category").agg(
        avg_fixed=("price", "mean"),
        avg_dynamic=("adjusted_price", "mean"),
        avg_margin_fixed=("margin_pct", "mean"),
        avg_margin_dynamic=("sim_margin", "mean"),
    ).round(2)
    cat_summary["price_lift_%"] = (
        (cat_summary["avg_dynamic"] - cat_summary["avg_fixed"])
        / cat_summary["avg_fixed"] * 100
    ).round(1)
    cat_summary["margin_lift_pp"] = (
        cat_summary["avg_margin_dynamic"] - cat_summary["avg_margin_fixed"]
    ).round(1)
    print(cat_summary[["avg_fixed", "avg_dynamic", "price_lift_%",
                        "avg_margin_fixed", "avg_margin_dynamic",
                        "margin_lift_pp"]].to_string())

    total_fixed_rev   = (df["price"] * df["purchase_count"]).sum()
    total_dynamic_rev = (df["adjusted_price"] * df["purchase_count"]).sum()
    rev_lift = (total_dynamic_rev - total_fixed_rev) / total_fixed_rev * 100

    print(f"\n── Revenue simulation ──")
    print(f"  Fixed pricing revenue  : ₹{total_fixed_rev:,.0f}")
    print(f"  Dynamic pricing revenue: ₹{total_dynamic_rev:,.0f}")
    print(f"  Estimated revenue lift : {rev_lift:+.1f}%")

    print("\n── Dynamic pricing rules (recommendation) ──")
    rules = {
        "inelastic (Beauty, Home)":          "Tăng +15–25% · Trigger: competitor gap >5% hoặc demand spike 1.2×",
        "elastic (Electronics, Books)":       "Giữ ±8% market · Volume discount khi cart >₹500 · Cap tăng +7%",
        "moderate (Clothing, Sports, Toys)":  "Tăng +10–14% · A/B test 5–10% traffic split mỗi tuần",
    }
    for seg, rule in rules.items():
        print(f"  [{seg}]")
        print(f"    → {rule}")

    print("\n" + "═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(data_path: str = DATA_PATH):
    print(f"Loading data from {data_path}...")
    df, elasticities = load_pipeline(data_path)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    run_sql_analysis(df)

    print("\nGenerating charts → charts/")
    plot_elasticity(df)
    plot_cr_vs_price(df)
    plot_revenue_parabola(df)
    plot_margin_by_bin(df)
    plot_cr_heatmap(df)
    plot_price_distribution(df)

    print_report(df, elasticities)
    return df


if __name__ == "__main__":
    run_analysis()
