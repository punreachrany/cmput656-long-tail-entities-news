import os
import json
import pandas as pd
from tqdm import tqdm

# 1. Load the mapping dictionary
mapping_csv_path = "tests/type_mapping_dictionary_st.csv"

if not os.path.exists(mapping_csv_path):
    print(f"❌ Error: Mapping file '{mapping_csv_path}' not found.")
    exit(1)

mapping_df = pd.read_csv(mapping_csv_path)
# Convert DataFrame to a standard Python dictionary for lightning-fast O(1) lookups
mapping_dict = dict(zip(mapping_df['el_fine_type'], mapping_df['mapped_label']))

processed_data = []

# 2. Iterate through all 50 JSONL files
print("Processing JSONL files...")
for i in tqdm(range(50)):
    file_path = f"unique_el/el_output_{i}.jsonl"
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. Skipping.")
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            # --- Handle Coarse Type ---
            coarse_type = record.get("el_coarse_type")
            if coarse_type is None:
                coarse_type = "NIL_CG"

            # --- Handle Fine Type ---
            fine_types = record.get("el_fine_types", [])
            extracted_fine_type = None
            if isinstance(fine_types, list) and len(fine_types) > 0:
                extracted_fine_type = fine_types[0]
            
            # Map the fine type if it exists, otherwise set to "NIL_FG"
            if extracted_fine_type is None:
                mapped_label = "NIL_FG"
            else:
                # Use .get() to find the map, fallback to keeping it as-is if no map is found
                mapped_label = mapping_dict.get(extracted_fine_type, extracted_fine_type)

            # 3. Construct the flattened row with renamed columns
            processed_row = {
                "ner_id": record.get("ner_id"),
                "el_id": record.get("el_id"),
                "ner_text": record.get("text"),                 # Renamed from 'text'
                "ner_type": record.get("ner_label"),            # Renamed from 'ner_label'
                "el_text": record.get("el_label"),              # Renamed from 'el_label'
                "el_coarse_type": coarse_type,                  # Handled null -> "NIL_CG"
                "el_fine_type": mapped_label                    # Handled null -> "NIL_FG"
            }
            
            processed_data.append(processed_row)

# 4. Convert to DataFrame and save to CSV
print("Converting to DataFrame and exporting...")
final_df = pd.DataFrame(processed_data)

# Ensure the results directory exists so it doesn't crash on save
os.makedirs("results", exist_ok=True)

output_file = "results/long_tail_result.csv"
final_df.to_csv(output_file, index=False)
print(f"\n✅ Successfully saved {len(final_df):,} records to '{output_file}'!")