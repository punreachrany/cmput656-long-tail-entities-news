import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR = "unique_refined_el"
OUTPUT_DIR = "tests/visualizations"
OUTPUT_COUNTS_CSV = f"{OUTPUT_DIR}/refined_summary_table_counts.csv"
OUTPUT_PCT_CSV = f"{OUTPUT_DIR}/refined_summary_table_percentages.csv"
NUM_FILES = 50

EL_TYPES = [
    "PERSON", "WORK_OF_ART", "GPE", "ORG", "FAC", "DATE", 
    "LANGUAGE", "CARDINAL", "PRODUCT", "EVENT", "PERCENT", 
    "TIME", "ORDINAL", "QUANTITY", "MONEY"
]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Data Generation ────────────────────────────────────────────────────────
def generate_summary_tables():
    stats = defaultdict(lambda: {col: 0 for col in ["Total"] + EL_TYPES + ["NIL", "Unlinked"]})
    
    print("Scanning files and aggregating data...")
    for i in tqdm(range(NUM_FILES), desc="Processing Files"):
        file_path = os.path.join(INPUT_DIR, f"el_output_refined_{i}.jsonl")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ner_type = obj.get("ner_type")
                    el_type = obj.get("el_type")
                    el_id = obj.get("el_id")
                    
                    if not ner_type: continue
                        
                    stats[ner_type]["Total"] += 1
                    
                    if el_id is None or el_id == "":
                        stats[ner_type]["Unlinked"] += 1
                    else:
                        if el_type is None or el_type == "":
                            stats[ner_type]["NIL"] += 1
                        elif el_type in EL_TYPES:
                            stats[ner_type][el_type] += 1
                except json.JSONDecodeError:
                    continue

    if not stats:
        raise ValueError("No data found! Check your INPUT_DIR.")

    # Format Counts Table
    rows = [{"ner_type": k, **v} for k, v in stats.items()]
    df_counts = pd.DataFrame(rows)
    columns_order = ["ner_type", "Total"] + EL_TYPES + ["NIL", "Unlinked"]
    
    for col in columns_order:
        if col not in df_counts.columns: df_counts[col] = 0
            
    df_counts = df_counts[columns_order].sort_values(by="Total", ascending=False).reset_index(drop=True)
    
    # Format Percentages Table
    df_pct = df_counts.copy()
    cols_to_convert = EL_TYPES + ["NIL", "Unlinked"]
    for col in cols_to_convert:
        df_pct[col] = ((df_pct[col] / df_pct["Total"]) * 100).round(2)
    
    df_counts.to_csv(OUTPUT_COUNTS_CSV, index=False)
    df_pct.to_csv(OUTPUT_PCT_CSV, index=False)
    
    print(f"✅ Tables saved to {OUTPUT_DIR}/")
    return df_counts, df_pct

# ── 2. Visualizations ─────────────────────────────────────────────────────────

def plot_heatmap(df_pct):
    print("Generating Heatmap...")
    plt.figure(figsize=(14, 10))
    # Drop 'Total' for the heatmap, set ner_type as the index
    plot_data = df_pct.set_index('ner_type').drop(columns=['Total'])
    
    sns.heatmap(plot_data, annot=False, cmap="YlGnBu", linewidths=.5)
    plt.title("GLiNER to ReFinED Type Mapping (%)", fontsize=16)
    plt.ylabel("GLiNER NER Type", fontsize=12)
    plt.xlabel("ReFinED Linking Outcome", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_mapping_heatmap.png", dpi=300)
    plt.close()

def plot_stacked_bar(df_pct):
    print("Generating Stacked Bar Chart...")
    plot_data = df_pct.sort_values('Total', ascending=True) # Ascending looks better for horizontal bars
    cols_to_plot = EL_TYPES + ["NIL", "Unlinked"]
    
    fig = px.bar(
        plot_data, 
        y="ner_type", 
        x=cols_to_plot, 
        orientation='h',
        title="100% Stacked Linking Outcomes by NER Type",
        labels={"value": "Percentage (%)", "variable": "Linking Outcome", "ner_type": "NER Type"},
        height=800
    )
    fig.write_html(f"{OUTPUT_DIR}/2_stacked_bar.html")

def plot_long_tail(df_pct):
    print("Generating Long-Tail Chart...")
    plt.figure(figsize=(12, 8))
    plot_data = df_pct.sort_values('Unlinked', ascending=False)
    
    sns.barplot(data=plot_data, x='Unlinked', y='ner_type', palette="Reds_r")
    plt.title("The Long-Tail: Percentage of Unlinked Entities per Type", fontsize=16)
    plt.xlabel("Unlinked Percentage (%)", fontsize=12)
    plt.ylabel("GLiNER NER Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_long_tail_unlinked.png", dpi=300)
    plt.close()

def plot_sankey(df_counts):
    print("Generating Sankey Diagram...")
    # Nodes: All NER Types + All EL Outcomes
    ner_nodes = df_counts['ner_type'].tolist()
    target_nodes = EL_TYPES + ["NIL", "Unlinked"]
    all_nodes = ner_nodes + target_nodes
    
    # Create mapping from name to index
    node_indices = {name: i for i, name in enumerate(all_nodes)}
    
    sources, targets, values = [], [], []
    
    # Populate links based on raw counts
    for _, row in df_counts.iterrows():
        source_idx = node_indices[row['ner_type']]
        for target_name in target_nodes:
            count = row[target_name]
            if count > 0:
                sources.append(source_idx)
                targets.append(node_indices[target_name])
                values.append(count)
                
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = all_nodes
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values
        )
    )])
    fig.update_layout(title_text="Entity Flow: Extraction to Linking", font_size=10, height=900)
    fig.write_html(f"{OUTPUT_DIR}/4_sankey_diagram.html")

def plot_scatter(df_counts, df_pct):
    print("Generating Scatter Plot...")
    plt.figure(figsize=(10, 6))
    
    # We need Total from counts and Unlinked from pct
    plot_data = pd.DataFrame({
        'ner_type': df_counts['ner_type'],
        'Total': df_counts['Total'],
        'Unlinked_Pct': df_pct['Unlinked']
    })
    
    sns.scatterplot(data=plot_data, x='Total', y='Unlinked_Pct', s=100, color="indigo", alpha=0.7)
    
    # Annotate points with their NER type
    for i in range(plot_data.shape[0]):
        plt.text(plot_data['Total'][i], plot_data['Unlinked_Pct'][i] + 1, 
                 plot_data['ner_type'][i], horizontalalignment='center', size='small')

    plt.xscale('log') # Log scale because total counts range from tiny to massive
    plt.title("Are Rare Entities Harder to Link?", fontsize=16)
    plt.xlabel("Total Extractions (Log Scale)", fontsize=12)
    plt.ylabel("Unlinked Entities (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_scatter_frequency.png", dpi=300)
    plt.close()

# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_counts, df_pct = generate_summary_tables()
    
    plot_heatmap(df_pct)
    plot_stacked_bar(df_pct)
    plot_long_tail(df_pct)
    plot_sankey(df_counts)
    plot_scatter(df_counts, df_pct)
    
    print(f"\n🎉 All done! Check the '{OUTPUT_DIR}' folder for your CSVs and charts.")