import argparse
import os
import pandas as pd

"""
# Default paths
python3 count_frequencies.py

# Custom paths
python3 count_frequencies.py \
  -i el_outputs/combined_el_output.csv \
  -o ner_to_el_frequencies.csv
"""


def determine_el_category(row):
    if not row["el_id"] or str(row["el_id"]).lower() in ["nan", "none"]:
        return "Unlinkable"
    if not row["el_type"] or str(row["el_type"]).lower() in ["nan", "none"]:
        return "NIL"
    return row["el_type"]


def count_type_frequencies(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, dtype=str)

    df["ner_type"] = df["ner_type"].fillna("UNKNOWN").str.strip()
    df["el_id"]    = df["el_id"].fillna("").str.strip()
    df["el_type"]  = df["el_type"].fillna("").str.strip()

    print("Categorizing EL types...")
    df["final_el_type"] = df.apply(determine_el_category, axis=1)

    print("Calculating frequencies...")
    freq_df = (
        df.groupby(["ner_type", "final_el_type"])
        .size()
        .reset_index(name="frequency")
        .sort_values(by=["ner_type", "frequency"], ascending=[True, False])
    )

    freq_df.to_csv(output_file, index=False)
    print(f"\nDone! Results saved to '{output_file}'")

    print("\n--- FULL FREQUENCY RESULTS ---")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(freq_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count NER-to-EL type frequencies from combined EL output."
    )
    parser.add_argument(
        "-i", "--input",
        default="el_outputs/combined_el_output.csv",
        help="Path to combined EL output CSV (default: el_outputs/combined_el_output.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="ner_to_el_frequencies.csv",
        help="Path to save the frequency report CSV (default: ner_to_el_frequencies.csv)"
    )
    args = parser.parse_args()

    count_type_frequencies(args.input, args.output)