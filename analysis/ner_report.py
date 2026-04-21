import os
import argparse
import pandas as pd
from constants import LABELS

"""
python3 generate_report.py -i outputs_gliner/combined_ner_output.csv -o outputs_gliner/combined_ner_report.csv
"""


def generate_label_report(input_file, output_file):
    print(f"Reading data from {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return

    df['label'] = df['label'].astype(str).str.strip().str.lower()

    label_counts = df['label'].value_counts().to_dict()

    report_data = [{"Category": "TOTAL_ENTRIES", "Count": len(df)}]

    for label in LABELS:
        clean_label = label.strip().lower()
        report_data.append({
            "Category": clean_label,
            "Count": label_counts.get(clean_label, 0)
        })

    report_df = pd.DataFrame(report_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    report_df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n--- NER Counts Report ---")
    print(report_df.to_string(index=False))
    print("-------------------------")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a label count report from GLiNER NER output.")
    parser.add_argument("-i", "--input",  required=True, help="Path to combined NER output CSV")
    parser.add_argument("-o", "--output", required=True, help="Path to save the report CSV")
    args = parser.parse_args()

    generate_label_report(args.input, args.output)