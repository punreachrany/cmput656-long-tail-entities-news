"""
NER × EL Type Mapping Table
Generates a crosstab: rows = ner_type, columns = el_fine_type (+ el_coarse_type), with TOTAL.

Usage:
    python make_type_table.py --input results/long_tail_result.csv
    python make_type_table.py --input results/long_tail_result.csv --out figures/
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

LABELS = [
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language", "date", "time",
    "percent", "money", "quantity", "ordinal", "cardinal", "religion",
    "political_party", "nationality", "ethnic_group", "title", "award",
    "disease", "chemical", "weapon", "vehicle", "currency", "brand",
]

# ── NIL note ──────────────────────────────────────────────────────────────────
# NIL_CG / NIL_FG = entity was processed by the EL system but Wikipedia
# returned no usable coarse / fine-grained type for it.
# It does NOT mean the entity could not be linked at all.
# ─────────────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df["ner_type"]       = df["ner_type"].str.lower().str.strip()
    df["el_fine_type"]   = df["el_fine_type"].str.lower().str.strip()
    df["el_coarse_type"] = df["el_coarse_type"].str.upper().str.strip()
    return df


def build_tables(df):
    """Return (fine_ct, coarse_ct) — raw count crosstabs with TOTAL column/row."""

    # ── Fine-grained table ────────────────────────────────────────────────────
    fine_ct = (
        pd.crosstab(df["ner_type"], df["el_fine_type"])
          .reindex(index=[l for l in LABELS if l in df["ner_type"].values])
          .fillna(0).astype(int)
    )
    fine_ct["TOTAL"] = fine_ct.sum(axis=1)
    fine_ct.loc["TOTAL"] = fine_ct.sum(axis=0)

    # ── Coarse table ──────────────────────────────────────────────────────────
    coarse_ct = (
        pd.crosstab(df["ner_type"], df["el_coarse_type"])
          .reindex(index=[l for l in LABELS if l in df["ner_type"].values])
          .fillna(0).astype(int)
    )
    coarse_ct["TOTAL"] = coarse_ct.sum(axis=1)
    coarse_ct.loc["TOTAL"] = coarse_ct.sum(axis=0)

    return fine_ct, coarse_ct


def save_csvs(fine_ct, coarse_ct, out):
    fine_ct.to_csv(os.path.join(out, "table_ner_x_el_fine.csv"))
    coarse_ct.to_csv(os.path.join(out, "table_ner_x_el_coarse.csv"))
    print("[✓] CSVs saved: table_ner_x_el_fine.csv  |  table_ner_x_el_coarse.csv")


def print_table(ct, title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    # pretty-print with aligned columns
    with pd.option_context("display.max_columns", 50,
                           "display.width",      200,
                           "display.float_format", "{:,.0f}".format):
        print(ct.to_string())
    print()


# ── Heatmap visualisation ─────────────────────────────────────────────────────

def _heatmap(ax, data, cmap, title, note=None):
    """Draw a single annotated heatmap on ax (excludes TOTAL row/col for colour scale)."""
    body = data.drop(index="TOTAL", columns="TOTAL", errors="ignore")
    vals = body.values.astype(float)
    vals_plot = np.where(vals == 0, np.nan, vals)   # mask zeros → white

    im = ax.imshow(vals_plot, aspect="auto", cmap=cmap,
                   norm=LogNorm(vmin=1, vmax=max(vals.max(), 1)))

    rows, cols = body.index.tolist(), body.columns.tolist()
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)

    # Cell annotations
    vmax = vals.max() if vals.max() > 0 else 1
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = vals[i, j]
            if v > 0:
                ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > vmax * 0.4 else "#222")

    # TOTAL column as a separate strip on the right
    if "TOTAL" in data.columns:
        tot = data.loc[rows, "TOTAL"].values.astype(float)
        for i, t in enumerate(tot):
            ax.text(len(cols) + 0.55, i, f"{int(t):,}",
                    ha="left", va="center", fontsize=6.5,
                    color="#333", fontweight="bold")

    ax.set_title(title, fontweight="bold", pad=8, fontsize=9)
    if note:
        ax.text(0.5, -0.22, note, transform=ax.transAxes,
                ha="center", fontsize=7, color="#666", style="italic")

    return im


def plot_heatmaps(fine_ct, coarse_ct, out):
    n_fine_rows   = len(fine_ct)   - 1   # exclude TOTAL row
    n_coarse_rows = len(coarse_ct) - 1

    fig_h = max(6, n_fine_rows * 0.36 + 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(16, fig_h),
                              gridspec_kw={"width_ratios": [3, 2], "wspace": 0.45})

    cmap_fine   = plt.cm.Blues
    cmap_coarse = plt.cm.Oranges

    nil_note = (
        "NIL_FG / NIL_CG = EL found the entity but Wikipedia returned no usable type  "
        "(not the same as 'unlinkable')"
    )

    im_f = _heatmap(axes[0], fine_ct,   cmap_fine,
                    "NER Type × EL Fine-Grained Type  (count)", note=nil_note)
    im_c = _heatmap(axes[1], coarse_ct, cmap_coarse,
                    "NER Type × EL Coarse Type  (count)")

    # Colorbars
    cb_f = fig.colorbar(im_f, ax=axes[0], fraction=0.025, pad=0.02, shrink=0.7)
    cb_f.set_label("Count (log scale)", fontsize=7)
    cb_c = fig.colorbar(im_c, ax=axes[1], fraction=0.04,  pad=0.02, shrink=0.7)
    cb_c.set_label("Count (log scale)", fontsize=7)

    # TOTAL header label
    for ax in axes:
        ax.text(1.04, 1.01, "TOTAL →", transform=ax.transAxes,
                fontsize=6.5, color="#333", ha="left")

    fig.suptitle(
        "Table — NER Type × EL Type Mapping\n"
        "(GLiNER → GENRE/Wikipedia, Signal-1M)",
        fontweight="bold", fontsize=11, y=1.01
    )

    path_png = os.path.join(out, "table_ner_x_el_heatmap.png")
    path_pdf = os.path.join(out, "table_ner_x_el_heatmap.pdf")
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Heatmap saved: table_ner_x_el_heatmap  (.png / .pdf)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/long_tail_result.csv")
    parser.add_argument("--out",   default="figures/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.input)

    fine_ct, coarse_ct = build_tables(df)

    print_table(fine_ct,   "NER Type × EL Fine-Grained Type (counts)")
    print_table(coarse_ct, "NER Type × EL Coarse Type (counts)")

    save_csvs(fine_ct, coarse_ct, args.out)
    plot_heatmaps(fine_ct, coarse_ct, args.out)


if __name__ == "__main__":
    main()