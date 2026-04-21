import os
import pandas as pd

# Import the LABELS list from your constants.py file
from constants import LABELS

def generate_label_report(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # 1. Get the total number of entries
    total_entries = len(df)
    
    # 2. Clean the label column just to be safe (lowercase and remove extra spaces)
    # We use lowercase here so it matches the format of your LABELS list perfectly.
    df['label'] = df['label'].astype(str).str.strip().str.lower()

    # 3. Count the occurrences of each label
    label_counts = df['label'].value_counts().to_dict()

    # 4. Format the data for the report
    # Start with the total entries
    report_data = [
        {"Category": "TOTAL_ENTRIES", "Count": total_entries}
    ]
    
    # Dynamically loop through every label in your constants file
    for label in LABELS:
        # Ensure the label from the list is also cleaned/lowercased to guarantee a match
        clean_label = label.strip().lower()
        report_data.append({
            "Category": clean_label, 
            "Count": label_counts.get(clean_label, 0)
        })

    # Convert to a DataFrame and save
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print the results to the terminal so you can see them immediately
    print("\n--- NER Counts Report ---")
    print(report_df.to_string(index=False))
    print("-------------------------")
    print(f"Success! Report saved to: {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    INPUT_FILE = "outputs_gliner/combined_ner_output.csv"
    OUTPUT_FILE = "outputs_gliner/combined_ner_report.csv"
    
    generate_label_report(INPUT_FILE, OUTPUT_FILE)