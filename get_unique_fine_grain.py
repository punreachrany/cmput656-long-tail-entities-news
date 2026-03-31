import pandas as pd
import json
import os
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR = "unique_el"
OUTPUT_FILE = "tests/master_fine_grain_types.csv"
NUM_FILES = 50 

def process_all_files():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Remove existing file to start fresh
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing {OUTPUT_FILE} to start fresh.")

    # ── Global Set for Uniqueness ──
    # We store strings like "ORG||Company" to track what we've already saved
    seen_pairs = set()
    first_file = True

    for i in range(NUM_FILES):
        file_path = os.path.join(INPUT_DIR, f"el_output_{i}.jsonl")
        
        if not os.path.exists(file_path):
            continue

        print(f"Processing: {file_path}...")
        
        chunk_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ner_label = obj.get("ner_label")
                    coarse = obj.get("el_coarse_type")
                    fine_types = obj.get("el_fine_types")

                    if fine_types and isinstance(fine_types, list):
                        for ft in fine_types:
                            if not ft: continue
                            
                            # Create a unique key for the pair
                            # We use || as a separator
                            pair_key = f"{ner_label}||{ft}"

                            if pair_key not in seen_pairs:
                                chunk_data.append({
                                    "ner_label": ner_label,
                                    "el_coarse_type": coarse,
                                    "el_fine_type": ft
                                })
                                # Mark as seen so we never add it again
                                seen_pairs.add(pair_key)
                                
                except (json.JSONDecodeError, KeyError):
                    continue

        if not chunk_data:
            continue

        # Convert the new unique pairs from this file to a DataFrame
        df_new_unique = pd.DataFrame(chunk_data)

        # Write to CSV in append mode
        df_new_unique.to_csv(OUTPUT_FILE, mode='a', index=False, header=first_file)
        
        first_file = False
        print(f"   Done. Added {len(df_new_unique):,} NEW unique pairs.")

    print(f"\n✅ Finished! Global unique pairs saved to: {OUTPUT_FILE}")
    print(f"   Total unique pairs found: {len(seen_pairs):,}")

if __name__ == "__main__":
    process_all_files()