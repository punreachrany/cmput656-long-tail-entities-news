import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("unique_el_final/combined_el_output_refined.csv", dtype=str)

# Drop rows where el_id is empty — no link, nothing to evaluate
df = df[df["el_id"].notna() & (df["el_id"].str.strip() != "")].copy()
print(f"Rows with el_id: {len(df):,}")

# ── Fine-grained → coarse NER mapping ────────────────────────────────────────
COARSE_NER_MAP = {
    "person":           "PERSON",
    "politician":       "PERSON",
    "religious_leader": "PERSON",
    "military_person":  "PERSON",
    "athlete":          "PERSON",
    "organization":     "ORGANIZATION",
    "political_party":  "ORGANIZATION",
    "website":          "ORGANIZATION",
    "sports_team":      "ORGANIZATION",
    "government_agency":"ORGANIZATION",
    "norp":             "ORGANIZATION",
    "location":         "LOCATION",
    "facility":         "LOCATION",
    "gpe":              "LOCATION",
}

# EL types → coarse EL bucket
def coarse_el(el_type):
    t = str(el_type).strip()
    if t == "PERSON":           return "PERSON"
    if t == "ORG":              return "ORGANIZATION"
    if t in {"GPE", "FAC"}:    return "LOCATION"
    return None   # ignore other EL types

# ── Prepare ───────────────────────────────────────────────────────────────────
df["coarse_ner"] = df["ner_type"].str.strip().map(COARSE_NER_MAP)
df["coarse_el"]  = df["el_type"].apply(coarse_el)

# Keep only rows in the 3 coarse NER types AND with a recognised coarse EL type
df = df[
    df["coarse_ner"].isin(["PERSON", "ORGANIZATION", "LOCATION"]) &
    df["coarse_el"].isin(["PERSON", "ORGANIZATION", "LOCATION"])
].copy()

print(f"Rows in scope: {len(df):,}")
print("\nCross-tab (coarse_ner × coarse_el) — available rows:")
print(pd.crosstab(df["coarse_ner"], df["coarse_el"]))

# ── Sample 50 per (coarse_ner, coarse_el) cell ───────────────────────────────
SAMPLE_N = 50
samples  = []

print()
for ner in ["PERSON", "ORGANIZATION", "LOCATION"]:
    for el in ["PERSON", "ORGANIZATION", "LOCATION"]:
        bucket  = df[(df["coarse_ner"] == ner) & (df["coarse_el"] == el)]
        n_avail = len(bucket)

        if n_avail == 0:
            print(f"  NER={ner} × EL={el}: NO DATA")
            continue

        n       = min(SAMPLE_N, n_avail)
        sampled = bucket.sample(n=n, random_state=42)
        samples.append(sampled)
        print(f"  NER={ner} × EL={el}: {n_avail:>6,} available → sampled {n}")

result = pd.concat(samples).reset_index(drop=True)

# ── Output columns ────────────────────────────────────────────────────────────
result = result[[
    "ner_id", "ner_text", "ner_type", "coarse_ner",
    "el_id",  "el_text",  "el_type",  "coarse_el",
    "ner_score"
]]

result["correct"] = ""   # annotator fills: Y / N
result["notes"]   = ""   # annotator fills: reason if wrong

result.to_csv("error_analysis_sample.csv", index=False, quoting=1)

print(f"\nTotal sampled : {len(result)} rows → error_analysis_sample.csv")
print("\nSample cross-tab:")
print(pd.crosstab(result["coarse_ner"], result["coarse_el"]))