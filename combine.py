import os
import glob
import json
import pandas as pd
import argparse

def combine_jsonl_to_csv(folder_path, output_filename):
    file_pattern = os.path.join(folder_path, 'ner_output_*.jsonl')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"No files matching 'ner_output_*.jsonl' were found in '{folder_path}'.")
        return

    print(f"Found {len(file_list)} JSONL files. Extracting and combining entities...")
    
    all_extracted_entities = []
    skipped_lines = 0
    
    for file in file_list:
        print(f"Processing {file}...")
        with open(file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    data = json.loads(line)
                    
                    doc_id = data.get('id')
                    entities = data.get('entities', [])
                    
                    for ent in entities:
                        all_extracted_entities.append({
                            'id': doc_id,
                            'text': ent.get('text'),
                            'label': ent.get('label'),
                            'start': ent.get('start'),
                            'end': ent.get('end'),
                            'score': ent.get('score')
                        })
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    continue
                        
    if all_extracted_entities:
        combined_df = pd.DataFrame(all_extracted_entities)
        combined_df = combined_df[['id', 'text', 'label', 'start', 'end', 'score']]
        
        output_path = os.path.join(folder_path, output_filename)
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\nSuccess! Combined data saved to: {output_path}")
        print(f"Total entity rows in combined file: {len(combined_df)}")
        if skipped_lines > 0:
            print(f"Note: Skipped {skipped_lines} corrupted lines across all files.")
    else:
        print("No entities were extracted from the files.")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Combine JSONL NER outputs into a single CSV.")
    
    # Add arguments for folder path and output file
    parser.add_argument(
        "-f", "--folder", 
        type=str, 
        required=True, 
        help="The folder path containing the 'ner_output_*.jsonl' files."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="combined_el_output.csv", 
        help="The name of the output CSV file (default: combined_el_output.csv)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the function with the parsed arguments
    combine_jsonl_to_csv(args.folder, args.output)