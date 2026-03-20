"""
reports/semantic_coverage_eda.py
---------------------------------
EDA focado em cobertura semantica: identifica o gap de metadados,
impacto nas interacoes, qualidade de texto e gera recomendacoes de estrategia.

Uso:
    python reports/semantic_coverage_eda.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
products = pd.read_parquet(PROCESSED_DIR / "products.parquet")

products_with_metadata = set(products["product_id"])
products_in_interactions = set(interactions["product_id"].unique())
covered = products_in_interactions & products_with_metadata
uncovered = products_in_interactions - products_with_metadata

interactions_covered = interactions[interactions["product_id"].isin(covered)]
interactions_uncovered = interactions[interactions["product_id"].isin(uncovered)]
prod_counts = interactions.groupby("product_id")["rating"].count().sort_values(ascending=False)
prod_uncovered_counts = interactions_uncovered.groupby("product_id")["rating"].count()

# ── Per-user coverage ──────────────────────────────────────────────────────
user_cov_pct = (
    interactions.groupby("user_id")["product_id"]
    .apply(lambda s: s.isin(covered).mean())
)

# ── Text length of covered products ───────────────────────────────────────
covered_prods = products[products["product_id"].isin(covered)].copy()

def _text_len(row: pd.Series) -> int:
    parts = []
    title = row["title"]
    if title is not None and not (isinstance(title, float) and pd.isna(title)):
        parts.append(str(title))
    desc = row["description"]
    if desc is not None and not (isinstance(desc, float) and pd.isna(desc)):
        if isinstance(desc, (list, np.ndarray)):
            text = " ".join(str(d) for d in desc)
        else:
            text = str(desc)
        if text.strip():
            parts.append(text)
    return len(" | ".join(parts))

covered_prods["text_len"] = covered_prods.apply(_text_len, axis=1)

# ── Build figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Semantic Coverage Analysis — SmartRec",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

colors_duo = ["#2ecc71", "#e74c3c"]

# A: catalog coverage donut
ax1 = fig.add_subplot(3, 3, 1)
sizes_a = [len(covered), len(uncovered)]
ax1.pie(
    sizes_a,
    labels=["With metadata\n(embeddable)", "No metadata\n(gap)"],
    colors=colors_duo,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.55),
)
ax1.set_title("Catalog: Unique Products", fontweight="bold")

# B: interactions coverage donut
ax2 = fig.add_subplot(3, 3, 2)
sizes_b = [len(interactions_covered), len(interactions_uncovered)]
ax2.pie(
    sizes_b,
    labels=["Covered", "Gap"],
    colors=colors_duo,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.55),
)
ax2.set_title("Interactions impacted by gap", fontweight="bold")

# C: top-K popular products embeddable %
ax3 = fig.add_subplot(3, 3, 3)
ks = [10, 25, 50, 100, 200, 500, 1000]
cov_pcts = [len(set(prod_counts.head(k).index) & covered) / k for k in ks]
ax3.plot(ks, [p * 100 for p in cov_pcts], "o-", color="steelblue", linewidth=2, markersize=7)
ax3.axhline(62.7, color="gray", linestyle="--", alpha=0.6, label="Avg 62.7%")
ax3.fill_between(
    ks,
    [p * 100 for p in cov_pcts],
    62.7,
    alpha=0.15,
    color="red",
    where=[p * 100 < 62.7 for p in cov_pcts],
)
ax3.set_xlabel("Top-K most popular products")
ax3.set_ylabel("% with embeddings")
ax3.set_title("Popular Products: % Embeddable", fontweight="bold")
ax3.set_ylim(0, 100)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# D: per-user coverage histogram
ax4 = fig.add_subplot(3, 3, 4)
ax4.hist(user_cov_pct * 100, bins=50, color="mediumpurple", edgecolor="white", alpha=0.85)
ax4.axvline(50, color="red", linestyle="--", linewidth=1.5, label="Median: 50%")
ax4.set_xlabel("% interactions with embeddings")
ax4.set_ylabel("Number of users")
ax4.set_title("Per-user semantic coverage", fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# E: gap products interaction distribution
ax5 = fig.add_subplot(3, 3, 5)
ax5.hist(
    prod_uncovered_counts,
    bins=60,
    color="#e67e22",
    edgecolor="white",
    log=True,
    alpha=0.85,
)
ax5.axvline(
    prod_uncovered_counts.median(),
    color="navy",
    linestyle="--",
    linewidth=1.5,
    label=f"Median: {prod_uncovered_counts.median():.0f}",
)
ax5.set_xlabel("Interactions per gap-product")
ax5.set_ylabel("Products (log scale)")
ax5.set_title("Gap products are NOT cold-start", fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# F: text quality of embeddable products
ax6 = fig.add_subplot(3, 3, 6)
ax6.hist(
    covered_prods["text_len"].clip(upper=3000),
    bins=60,
    color="#27ae60",
    edgecolor="white",
    alpha=0.85,
)
ax6.axvline(
    covered_prods["text_len"].median(),
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Median: {covered_prods['text_len'].median():.0f} chars",
)
ax6.set_xlabel("Text length (chars, clipped at 3000)")
ax6.set_ylabel("Products")
ax6.set_title("Text Quality (embeddable products)", fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.3)

# G: users by coverage bucket
ax7 = fig.add_subplot(3, 3, 7)
bins_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
bins_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
user_cov_bins = pd.cut(
    user_cov_pct, bins=bins_vals, labels=bins_labels, include_lowest=True
)
bin_counts = user_cov_bins.value_counts().sort_index()
bar_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
ax7.bar(bin_counts.index, bin_counts.values, color=bar_colors, edgecolor="white")
ax7.set_xlabel("User semantic coverage bucket")
ax7.set_ylabel("Number of users")
ax7.set_title("Users by semantic coverage bucket", fontweight="bold")
ax7.grid(True, alpha=0.3, axis="y")
for i, val in enumerate(bin_counts.values):
    ax7.text(i, val + 200, f"{val:,}", ha="center", fontsize=8)

# H: category distribution of covered products
ax8 = fig.add_subplot(3, 3, 8)
cat_counts = covered_prods["category"].value_counts().head(8)
ax8.barh(cat_counts.index[::-1], cat_counts.values[::-1], color="#3498db", edgecolor="white")
ax8.set_xlabel("Products")
ax8.set_title("Categories (covered products)", fontweight="bold")
ax8.grid(True, alpha=0.3, axis="x")

# I: summary table
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
table_data = [
    ["Metric", "Value", "Implication"],
    ["Unique products w/ metadata", "9,321 (62.7%)", "Semantic ceiling"],
    ["Gap products (no metadata)", "5,555 (37.3%)", "CF-only fallback"],
    ["Interactions covered", "385K (50.0%)", "Hybrid alpha max 0.5"],
    ["Top-100 prods covered", "26 (26%)", "CF primary for popular"],
    ["Median user coverage", "50%", "Most users are hybrid"],
    ["Users with 0% coverage", "2.5%", "Pure CF path needed"],
    ["Gap products median ints", "28", "Not cold-start"],
    ["Products w/ empty desc", "850 (9.1%)", "Title-only embed OK"],
    ["Recoverable from raw", "0 (0%)", "Gap is permanent"],
]
the_table = ax9.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="left",
    loc="center",
    colWidths=[0.42, 0.28, 0.30],
)
the_table.auto_set_font_size(False)
the_table.set_fontsize(8)
the_table.scale(1, 1.4)
ax9.set_title("Key Metrics & Implications", fontweight="bold", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = FIGURES_DIR / "semantic_coverage_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
