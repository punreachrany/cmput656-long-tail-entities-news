import pandas as pd
from tqdm import tqdm

# =========================
# Load data
# =========================
print("loading dataframes...")
ner_df = pd.read_csv("combined_unique_ner.csv")
el_df  = pd.read_csv("combined_unique_el.csv")

# =========================
# Normalize text
# =========================
ner_df["norm_text"] = ner_df["text"].str.strip().str.lower()
el_df["norm_text"]  = el_df["text"].str.strip().str.lower()

# =========================
# Clean column names
# =========================
ner_df = ner_df.rename(columns={
    "id": "ner_id",
    "label": "ner_label",
    "score": "ner_score"
})

# =========================
# Deduplicate EL (IMPORTANT FIX)
# Keep highest score per norm_text
# =========================
print("🔨 Building EL lookup index (deduplicated by best score)...")

el_df = el_df.sort_values("score", ascending=False)
el_df_unique = el_df.drop_duplicates(subset=["norm_text"], keep="first")

# Build lookup dictionary
el_lookup = el_df_unique.set_index("norm_text")[["article_id", "label", "score"]].to_dict("index")

# =========================
# Mapping
# =========================
el_ids = []
el_labels = []
el_scores = []

print(f"🚀 Mapping {len(ner_df):,} entities...")

for text in tqdm(ner_df["norm_text"], desc="Mapping NER → EL", unit="row"):
    match = el_lookup.get(text)

    if match:
        el_ids.append(match["article_id"])
        el_labels.append(match["label"])
        el_scores.append(match["score"])
    else:
        el_ids.append("NA")
        el_labels.append("NA")
        el_scores.append("NA")

# =========================
# Attach results
# =========================
ner_df["el_article_id"] = el_ids
ner_df["el_label"] = el_labels
ner_df["el_score"] = el_scores

# =========================
# Rename final columns
# =========================
final_df = ner_df[[
    "text",
    "ner_id",
    "el_article_id",
    "ner_label",
    "el_label",
    "ner_score",
    "el_score"
]]

# =========================
# Fill missing values
# =========================
final_df = final_df.fillna("NA")

# =========================
# Mapping statistics
# =========================
total = len(final_df)
matched = (final_df["el_article_id"] != "NA").sum()
mapping_pct = (matched / total) * 100

print("\n📊 Mapping Completion:")
print(f"Total:   {total:,}")
print(f"Matched: {matched:,} ({mapping_pct:.2f}%)")

# =========================
# Save output
# =========================
output_file = "ner_el_mapped.csv"
print(f"\n💾 Saving to {output_file} ...")
final_df.to_csv(output_file, index=False)

print("✅ Done!")