import argparse
import pandas as pd

"""
python3 error_analysis_sample.py

# Custom paths
python3 error_analysis_sample.py \
  -i el_outputs/combined_el_output.csv \
  -o error_analysis_sample.csv \
  -n 50
"""

# ── Coarse NER mapping ────────────────────────────────────────────────────────
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

# EL type → coarse EL bucket
def coarse_el(el_type):
    t = str(el_type).strip()
    if t == "PERSON":        return "PERSON"
    if t == "ORG":           return "ORGANIZATION"
    if t in {"GPE", "FAC"}:  return "LOCATION"
    return None


def main(input_file, output_file, sample_n):
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, dtype=str)

    # Drop rows with no el_id
    df = df[df["el_id"].notna() & (df["el_id"].str.strip() != "")].copy()
    print(f"Rows with el_id: {len(df):,}")

    # Map coarse types
    df["coarse_ner"] = df["ner_type"].str.strip().map(COARSE_NER_MAP)
    df["coarse_el"]  = df["el_type"].apply(coarse_el)

    # Keep only rows in scope
    df = df[
        df["coarse_ner"].isin(["PERSON", "ORGANIZATION", "LOCATION"]) &
        df["coarse_el"].isin(["PERSON", "ORGANIZATION", "LOCATION"])
    ].copy()

    print(f"Rows in scope: {len(df):,}")
    print("\nAvailable rows per cell (coarse_ner × coarse_el):")
    print(pd.crosstab(df["coarse_ner"], df["coarse_el"]))

    # ── Stratified sample ─────────────────────────────────────────────────────
    samples = []
    print()
    for ner in ["PERSON", "ORGANIZATION", "LOCATION"]:
        for el in ["PERSON", "ORGANIZATION", "LOCATION"]:
            bucket  = df[(df["coarse_ner"] == ner) & (df["coarse_el"] == el)]
            n_avail = len(bucket)

            if n_avail == 0:
                print(f"  NER={ner} × EL={el}: NO DATA")
                continue

            n       = min(sample_n, n_avail)
            sampled = bucket.sample(n=n, random_state=42)
            samples.append(sampled)
            print(f"  NER={ner} × EL={el}: {n_avail:>6,} available → sampled {n}")

    result = pd.concat(samples).reset_index(drop=True)

    # ── Output ────────────────────────────────────────────────────────────────
    result = result[[
        "ner_id", "ner_text", "ner_type", "coarse_ner",
        "el_id",  "el_text",  "el_type",  "coarse_el",
        "ner_score"
    ]]
    result["correct"] = ""   # annotator fills: Both Correct / NER Correct / EL Correct
    result["notes"]   = ""   # annotator fills: reason if wrong

    result.to_csv(output_file, index=False, quoting=1)

    print(f"\nTotal sampled: {len(result)} rows → {output_file}")
    print("\nSample cross-tab:")
    print(pd.crosstab(result["coarse_ner"], result["coarse_el"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified error analysis sample from EL output (3x3 NER x EL grid)."
    )
    parser.add_argument(
        "-i", "--input",
        default="el_outputs/combined_el_output.csv",
        help="Path to combined EL output CSV (default: el_outputs/combined_el_output.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="error_analysis_sample.csv",
        help="Path to save the sampled output CSV (default: error_analysis_sample.csv)"
    )
    parser.add_argument(
        "-n", "--sample-n",
        type=int,
        default=50,
        help="Number of entities to sample per NER x EL cell (default: 50)"
    )
    args = parser.parse_args()

    main(args.input, args.output, args.sample_n)