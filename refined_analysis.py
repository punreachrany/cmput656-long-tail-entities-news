"""
================================================================================
NER/EL Pipeline Analysis Script (Hybrid PNG/HTML Version)
Project: A Fine-Grained Study of Long-Tail Entities in News Articles
Author context: Replicating/extending Esquivel et al. (2017) methodology
                using GLiNER (NER) + ReFinED (EL) instead of Stanford NER +
                DBPedia Spotlight.

This script:
  Goal 1 – Parses 50 JSONL files and builds granular count/percentage tables
  Goal 2 – Macro-level aggregation into 5 categories
  Goal 3 – 6 core visualizations (3 static PNGs, 3 interactive HTMLs)
  Goal 4 – 1 novel visualization (NER score × linking outcome analysis)

Usage:
    python analyze_el_pipeline.py

Expected directory layout:
    unique_refined_el/
        el_output_refined_0.jsonl
        ...
        el_output_refined_49.jsonl
    visualizations/          ← created automatically
================================================================================
"""

import os
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Plotly for interactive HTML exports
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# 30 GLiNER NER labels (canonical lower-case)
NER_TYPES = [
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language", "date", "time",
    "percent", "money", "quantity", "ordinal", "cardinal", "religion",
    "political_party", "nationality", "ethnic_group", "title", "award",
    "disease", "chemical", "weapon", "vehicle", "currency", "brand",
]

# 15 ReFinED EL coarse types (exact strings used in the data)
EL_TYPES = [
    "PERSON", "WORK_OF_ART", "GPE", "ORG", "FAC", "DATE", "LANGUAGE",
    "CARDINAL", "PRODUCT", "EVENT", "PERCENT", "TIME", "ORDINAL",
    "QUANTITY", "MONEY",
]

# Ordered column layout for the output CSVs
COUNT_COLS  = ["ner_type", "Total"] + EL_TYPES + ["NIL", "Unlinked"]
PCTAGE_COLS = COUNT_COLS  # same layout

DATA_DIR      = Path("unique_refined_el")
VIZ_DIR       = Path("visualizations_refined")
VIZ_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PARSE & AGGREGATE
# ──────────────────────────────────────────────────────────────────────────────

def parse_jsonl_files(data_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_records: list[dict] = []

    for i in range(50):
        fpath = data_dir / f"el_output_refined_{i}.jsonl"
        if not fpath.exists():
            print(f"  [WARN] Missing file: {fpath} – skipping.")
            continue

        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ner_type = (rec.get("ner_type") or "").strip().lower()
                if not ner_type: continue 

                el_id   = rec.get("el_id")   or ""
                el_type = (rec.get("el_type") or "").strip().upper()

                counts[ner_type]["Total"] += 1

                if not el_id:
                    counts[ner_type]["Unlinked"] += 1
                elif not el_type:
                    counts[ner_type]["NIL"] += 1
                elif el_type in EL_TYPES:
                    counts[ner_type][el_type] += 1
                else:
                    counts[ner_type]["NIL"] += 1

                raw_records.append({
                    "ner_type"  : ner_type,
                    "el_id"     : el_id,
                    "el_type"   : el_type,
                    "ner_score" : rec.get("ner_score"),
                    "outcome"   : (
                        "Unlinked" if not el_id else
                        "NIL"      if not el_type else
                        el_type
                    ),
                })

    rows = []
    for ner_type in set(NER_TYPES).union(counts.keys()):
        row = {"ner_type": ner_type, "Total": counts[ner_type]["Total"]}
        for col in EL_TYPES + ["NIL", "Unlinked"]:
            row[col] = counts[ner_type].get(col, 0)
        rows.append(row)

    counts_df = pd.DataFrame(rows, columns=COUNT_COLS)
    counts_df = counts_df.sort_values("Total", ascending=False).reset_index(drop=True)

    return counts_df, raw_records


def build_percentage_df(counts_df: pd.DataFrame) -> pd.DataFrame:
    pct_df = counts_df.copy()
    numeric_cols = [c for c in COUNT_COLS if c not in ("ner_type", "Total")]
    for col in numeric_cols:
        pct_df[col] = pct_df.apply(
            lambda r: round(r[col] / r["Total"] * 100, 2) if r["Total"] > 0 else 0.0,
            axis=1,
        )
    return pct_df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MACRO AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

NER_TO_MACRO: dict[str, str] = {
    "person": "PERSON", "title": "PERSON",
    "organization": "ORGANIZATION", "norp": "ORGANIZATION", "political_party": "ORGANIZATION", "brand": "ORGANIZATION",
    "gpe": "LOCATION", "location": "LOCATION", "facility": "LOCATION", "nationality": "LOCATION",
    "product": "MISC", "event": "MISC", "work_of_art": "MISC", "law": "MISC", "language": "MISC",
    "date": "MISC", "time": "MISC", "percent": "MISC", "money": "MISC", "quantity": "MISC",
    "ordinal": "MISC", "cardinal": "MISC", "religion": "MISC", "ethnic_group": "MISC",
    "award": "MISC", "disease": "MISC", "chemical": "MISC", "weapon": "MISC", "vehicle": "MISC", "currency": "MISC",
}

EL_TO_MACRO: dict[str, str] = {
    "PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "LOCATION", "FAC": "LOCATION",
    "WORK_OF_ART": "MISC", "DATE": "MISC", "LANGUAGE": "MISC", "CARDINAL": "MISC",
    "PRODUCT": "MISC", "EVENT": "MISC", "PERCENT": "MISC", "TIME": "MISC",
    "ORDINAL": "MISC", "QUANTITY": "MISC", "MONEY": "MISC",
    "NIL": "NIL/UNLINKED", "Unlinked": "NIL/UNLINKED",
}

MACRO_CATS = ["PERSON", "ORGANIZATION", "LOCATION", "MISC", "NIL/UNLINKED"]

def build_macro_df(raw_records: list[dict]) -> pd.DataFrame:
    macro_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in raw_records:
        ner_macro = NER_TO_MACRO.get(rec["ner_type"], "MISC")
        el_macro  = EL_TO_MACRO.get(rec["outcome"], "MISC")
        macro_counts[ner_macro]["Total"]    += 1
        macro_counts[ner_macro][el_macro]   += 1

    rows = []
    for cat in MACRO_CATS:
        row = {"macro_ner": cat, "Total": macro_counts[cat]["Total"]}
        for mc in MACRO_CATS:
            row[mc] = macro_counts[cat].get(mc, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    for mc in MACRO_CATS:
        df[mc] = df.apply(
            lambda r: round(r[mc] / r["Total"] * 100, 2) if r["Total"] > 0 else 0.0,
            axis=1,
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.9)
CMAP_DIV  = "RdYlGn"
FIG_DPI   = 150

def plot_mapping_heatmap(pct_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 1 (PNG) – Heatmap of ner_type × el_type outcomes."""
    heat_cols = EL_TYPES + ["NIL", "Unlinked"]
    heat_data = pct_df.set_index("ner_type")[heat_cols]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heat_data, annot=True, fmt=".1f", cmap=CMAP_DIV, linewidths=0.4, linecolor="white", cbar_kws={"label": "Row-wise %"}, ax=ax)
    ax.set_title("GLiNER NER Type → ReFinED EL Outcome  (% of row total)", fontsize=13)
    ax.set_xlabel("EL Outcome / Type", fontsize=11)
    ax.set_ylabel("GLiNER NER Type", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "viz1_mapping_heatmap.png", dpi=FIG_DPI)
    plt.close(fig)
    print("  [saved] viz1_mapping_heatmap.png")


def plot_stacked_bar_html(pct_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 2 (HTML) – Horizontal 100% stacked bar chart using Plotly."""
    plot_cols = EL_TYPES + ["NIL", "Unlinked"]
    df = pct_df[["ner_type"] + plot_cols].copy()
    df = df.sort_values("Unlinked", ascending=True)

    fig = px.bar(
        df, 
        y="ner_type", 
        x=plot_cols, 
        orientation='h',
        title="100% Stacked Horizontal Bar – EL Outcome Breakdown by NER Type",
        labels={"value": "Percentage (%)", "variable": "EL Outcome", "ner_type": "NER Type"},
        color_discrete_sequence=px.colors.qualitative.Prism,
        height=800
    )
    
    fig.update_layout(barmode='stack', hovermode='y unified')
    fig.write_html(str(out_dir / "viz2_stacked_bar.html"))
    print("  [saved] viz2_stacked_bar.html (Interactive)")


def plot_long_tail_bar(pct_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 3 (PNG) – Long-Tail bar chart: Unlinked % per ner_type."""
    df = pct_df[["ner_type", "Unlinked"]].sort_values("Unlinked", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df["ner_type"], df["Unlinked"], color=sns.color_palette("Reds_r", len(df)))
    ax.set_xlabel("GLiNER NER Type", fontsize=11)
    ax.set_ylabel("Unlinked %", fontsize=11)
    ax.set_title("The Long-Tail: Unlinked Percentage per NER Type (sorted desc.)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, df["Unlinked"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / "viz3_long_tail_bar.png", dpi=FIG_DPI)
    plt.close(fig)
    print("  [saved] viz3_long_tail_bar.png")


def plot_sankey_html(counts_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 4 (HTML) – Sankey diagram using Plotly."""
    link_src, link_tgt, link_val, link_lbl = [], [], [], []
    outcome_cols = EL_TYPES + ["NIL", "Unlinked"]
    all_nodes = list(counts_df["ner_type"]) + outcome_cols
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    for _, row in counts_df.iterrows():
        for oc in outcome_cols:
            val = row[oc]
            if val > 0:
                link_src.append(node_idx[row["ner_type"]])
                link_tgt.append(node_idx[oc])
                link_val.append(val)
                link_lbl.append(f"{row['ner_type']} → {oc}: {val}")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=15, line=dict(color="black", width=0.3),
            label=all_nodes,
            color=["#aec6cf"] * len(counts_df) + ["#ffb347"] * len(outcome_cols),
        ),
        link=dict(source=link_src, target=link_tgt, value=link_val, label=link_lbl),
    ))
    fig.update_layout(
        title_text="Sankey: GLiNER NER Type → ReFinED EL Outcome (raw counts)",
        font_size=12,
        height=900
    )
    fig.write_html(str(out_dir / "viz4_sankey.html"))
    print("  [saved] viz4_sankey.html (Interactive)")


def plot_freq_vs_linking_html(counts_df: pd.DataFrame, pct_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 5 (HTML) – Scatter Plot using Plotly."""
    df = counts_df[["ner_type", "Total"]].merge(pct_df[["ner_type", "Unlinked"]], on="ner_type")
    df = df[df["Total"] > 0]

    fig = px.scatter(
        df, x="Total", y="Unlinked", 
        text="ner_type",           # <-- ADDED: Tells Plotly to use the name as a permanent label
        hover_name="ner_type",
        log_x=True, 
        size_max=15,
        color="Unlinked", 
        color_continuous_scale="RdYlGn_r",
        title="Frequency vs Linking Failure Rate per NER Type (Interactive)",
        labels={"Total": "Total Entity Mentions (log scale)", "Unlinked": "Unlinked Percentage (%)"}
    )
    
    # <-- UPDATED: Formats the markers and pushes the text right above the dot so they don't overlap
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
        textposition='top center', 
        textfont=dict(size=10, color='black')
    )
    
    fig.update_layout(height=700)
    fig.write_html(str(out_dir / "viz5_freq_vs_linking.html"))
    print("  [saved] viz5_freq_vs_linking.html (Interactive)")


def plot_macro_stacked_bar(macro_df: pd.DataFrame, out_dir: Path) -> None:
    """Viz 6 (PNG) – 100% stacked bar for the 5 macro-categories."""
    df = macro_df.set_index("macro_ner")[MACRO_CATS]

    palette = sns.color_palette("Set2", len(MACRO_CATS))
    fig, ax = plt.subplots(figsize=(10, 6))
    lefts = np.zeros(len(df))
    for i, col in enumerate(MACRO_CATS):
        vals = df[col].values
        ax.bar(df.index, vals, bottom=lefts, color=palette[i], label=col)
        for j, (v, l) in enumerate(zip(vals, lefts)):
            if v > 3:
                ax.text(j, l + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        lefts += vals

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel("Macro NER Category", fontsize=11)
    ax.set_title("Macro-Level 100% Stacked Bar – EL Outcomes by NER Category", fontsize=11)
    ax.legend(title="Macro EL Outcome", loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "viz6_macro_stacked_bar.png", dpi=FIG_DPI)
    plt.close(fig)
    print("  [saved] viz6_macro_stacked_bar.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  NOVEL VISUALIZATION (Goal 4)
# ──────────────────────────────────────────────────────────────────────────────

def plot_novel_score_distribution(raw_records: list[dict], out_dir: Path) -> None:
    """Viz 7 (PNG) - NER Confidence Score Distribution by Linking Outcome."""
    df = pd.DataFrame([
        {
            "ner_type"   : r["ner_type"],
            "ner_score"  : r["ner_score"],
            "outcome_grp": ("Unlinked" if r["outcome"] == "Unlinked" else "NIL" if r["outcome"] == "NIL" else "Linked"),
            "macro_ner"  : NER_TO_MACRO.get(r["ner_type"], "MISC"),
        }
        for r in raw_records if r["ner_score"] is not None
    ])

    if df.empty:
        print("  [WARN] No ner_score data available; skipping novel plot.")
        return

    df["ner_score"] = pd.to_numeric(df["ner_score"], errors="coerce")
    df = df.dropna(subset=["ner_score"])

    outcome_palette = {"Linked": "#2ecc71", "NIL": "#f39c12", "Unlinked": "#e74c3c"}
    macro_order = ["PERSON", "ORGANIZATION", "LOCATION", "MISC", "NIL/UNLINKED"]
    macro_order_ner = [m for m in macro_order if m != "NIL/UNLINKED"]
    df = df[df["macro_ner"].isin(macro_order_ner)]

    fig, axes = plt.subplots(1, len(macro_order_ner), figsize=(16, 6), sharey=True)

    for ax, macro in zip(axes, macro_order_ner):
        sub = df[df["macro_ner"] == macro]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(data=sub, x="outcome_grp", y="ner_score", order=["Linked", "NIL", "Unlinked"], palette=outcome_palette, inner=None, linewidth=0.8, ax=ax)
        sample = sub.sample(min(500, len(sub)), random_state=42)
        sns.stripplot(data=sample, x="outcome_grp", y="ner_score", order=["Linked", "NIL", "Unlinked"], palette=outcome_palette, size=1.5, alpha=0.35, jitter=True, ax=ax)
        
        for i, outcome in enumerate(["Linked", "NIL", "Unlinked"]):
            med = sub[sub["outcome_grp"] == outcome]["ner_score"].median()
            if not math.isnan(med):
                ax.hlines(med, i - 0.35, i + 0.35, colors="black", linewidths=1.5, linestyles="--", label=f"median={med:.2f}" if i == 0 else "")
                ax.text(i, med + 0.005, f"{med:.2f}", ha="center", va="bottom", fontsize=7.5, color="black", fontweight="bold")

        ax.set_title(macro, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("GLiNER NER Score (confidence)", fontsize=10)
    fig.suptitle("Novel Plot: NER Extraction Confidence vs EL Linking Outcome\n(Violin + Strip, faceted by Macro NER Category)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "viz7_novel_score_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] viz7_novel_score_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  NER/EL Pipeline Analysis")
    print("=" * 70)

    print("\n[1/4] Parsing JSONL files …")
    counts_df, raw_records = parse_jsonl_files(DATA_DIR)
    print(f"      {len(raw_records):,} records loaded across {len(counts_df)} NER types.")

    print("\n[2/4] Building count & percentage tables …")
    pct_df = build_percentage_df(counts_df)
    counts_df.to_csv("refined_summary_table_counts.csv", index=False)
    pct_df.to_csv("refined_summary_table_percentages.csv", index=False)

    print("\n[3/4] Macro-level aggregation …")
    macro_df = build_macro_df(raw_records)
    macro_df.to_csv("macro_summary_percentages.csv", index=False)

    print("\n[4/4] Generating visualizations …")
    plot_mapping_heatmap(pct_df, VIZ_DIR)
    plot_stacked_bar_html(pct_df, VIZ_DIR)                # <-- NOW HTML
    plot_long_tail_bar(pct_df, VIZ_DIR)
    plot_sankey_html(counts_df, VIZ_DIR)                  # <-- NOW HTML
    plot_freq_vs_linking_html(counts_df, pct_df, VIZ_DIR) # <-- NOW HTML
    plot_macro_stacked_bar(macro_df, VIZ_DIR)
    plot_novel_score_distribution(raw_records, VIZ_DIR)

    print("\n" + "=" * 70)
    print("  All outputs written.")
    print(f"  Visualizations: {VIZ_DIR}/  (4 PNGs, 3 HTMLs)")
    print("=" * 70)


if __name__ == "__main__":
    main()