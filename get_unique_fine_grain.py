import pandas as pd
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR = "unique_refined_el"
OUTPUT_FILE = "tests/refined_types.csv"
NUM_FILES = 50 

def process_all_files():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Remove existing file to start fresh
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing {OUTPUT_FILE} to start fresh.")

    # ── Global Set for Uniqueness ──
    # We store strings like "PERSON", "ORG", etc., to track what we've seen
    seen_types = set()
    first_file = True

    for i in range(NUM_FILES):
        file_path = os.path.join(INPUT_DIR, f"el_output_refined_{i}.jsonl")
        
        if not os.path.exists(file_path):
            continue

        print(f"Processing: {file_path}...")
        
        chunk_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    
                    # Grab just the el_type
                    el_type = obj.get("el_type")

                    # Only process if it exists (ignoring null/empty ones)
                    if el_type:
                        if el_type not in seen_types:
                            chunk_data.append({
                                "el_type": el_type
                            })
                            # Mark as seen so we never add it again
                            seen_types.add(el_type)
                                
                except (json.JSONDecodeError, AttributeError):
                    continue

        if not chunk_data:
            continue

        # Convert the new unique types from this file to a DataFrame
        df_new_unique = pd.DataFrame(chunk_data)

        # Write to CSV in append mode
        df_new_unique.to_csv(OUTPUT_FILE, mode='a', index=False, header=first_file)
        
        # After the first successful write, we no longer need to write the CSV headers
        first_file = False
        print(f"   Done. Added {len(df_new_unique):,} NEW unique types.")

    print(f"\n✅ Finished! Global unique types saved to: {OUTPUT_FILE}")
    print(f"   Total unique types found: {len(seen_types):,}")

if __name__ == "__main__":
    process_all_files()