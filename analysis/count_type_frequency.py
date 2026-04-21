import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = "unique_el_final/combined_el_output_refined.csv"
OUTPUT_FILE = "ner_to_el_frequencies.csv"

def count_type_frequencies():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, dtype=str)
    
    # Clean the data to prevent mismatched grouping
    df['ner_type'] = df['ner_type'].fillna("UNKNOWN").str.strip()
    df['el_id'] = df['el_id'].fillna("").str.strip()
    df['el_type'] = df['el_type'].fillna("").str.strip()
    
    # --- APPLY CLASSIFICATION LOGIC ---
    def determine_el_category(row):
        # 1. Unlinkable: el_id is completely empty
        if not row['el_id'] or str(row['el_id']).lower() in ['nan', 'none']:
            return "Unlinkable"
        
        # 2. NIL: el_id exists, but el_type is empty
        if not row['el_type'] or str(row['el_type']).lower() in ['nan', 'none']:
            return "NIL"
            
        # 3. Otherwise, return the actual EL tag (CARDINAL, PERSON, ORG, etc.)
        return row['el_type']

    print("Categorizing EL types...")
    df['final_el_type'] = df.apply(determine_el_category, axis=1)
    
    # --- FREQUENCY LIST ---
    print("Calculating frequencies...")
    # Group by both columns and count the size of each group
    frequency_df = df.groupby(['ner_type', 'final_el_type']).size().reset_index(name='frequency')
    
    # Sort the results alphabetically by NER type, and then by highest frequency
    frequency_df = frequency_df.sort_values(by=['ner_type', 'frequency'], ascending=[True, False])
    
    # Save the result to a CSV
    frequency_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Done! Full results saved to '{OUTPUT_FILE}'")
    
    print("\n--- FULL FREQUENCY RESULTS ---")
    # Force pandas to print every single row without truncating
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(frequency_df.to_string(index=False))

if __name__ == "__main__":
    count_type_frequencies()