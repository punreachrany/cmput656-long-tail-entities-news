import os
import pandas as pd

# Import your labels list
from constants import LABELS

def subsample_entities(input_file, output_file, sample_size=200):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # 1. Clean column names (forces 'Label' to 'label' just in case)
    df.columns = df.columns.str.lower()

    # 2. Clean the label data to ensure exact matching
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    valid_labels = [label.strip().lower() for label in LABELS]

    # 3. Filter the dataframe to ONLY include rows with valid labels
    df_filtered = df[df['label'].isin(valid_labels)]
    
    # Safety check in case the filter drops everything
    if df_filtered.empty:
        print("Error: No entities found that match the labels in constants.py.")
        return

    print(f"Subsampling up to {sample_size} entities per label...")
    
    # 4. FOOLPROOF SAMPLING: Loop through each group instead of using .apply()
    samples = []
    for label_name, group in df_filtered.groupby('label'):
        n_to_sample = min(len(group), sample_size)
        sampled_group = group.sample(n=n_to_sample, random_state=42)
        samples.append(sampled_group)

    # Combine all the individual sampled chunks back into one DataFrame
    subsampled_df = pd.concat(samples, ignore_index=True)

    # 5. Keep ONLY the 'text' and 'label' columns
    final_df = subsampled_df[['text', 'label']]

    # 6. Save to the new CSV
    final_df.to_csv(output_file, index=False, encoding='utf-8')

    # Print summary
    print("\n--- Subsampling Summary ---")
    print(final_df['label'].value_counts().to_string())
    print("---------------------------")
    print(f"Total rows in new file: {len(final_df)}")
    print(f"Success! Saved to: {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    INPUT_FILE = "outputs_gliner/unique_ner_output.csv"
    OUTPUT_FILE = "outputs_gliner/subsampled_ner_50.csv"
    
    subsample_entities(INPUT_FILE, OUTPUT_FILE, sample_size=200)