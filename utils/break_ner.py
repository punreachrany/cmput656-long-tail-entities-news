import pandas as pd
import os
import math

def split_csv(input_file, output_folder, num_files=50):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Count total rows to determine split size
    print("Counting total rows in source file...")
    # Using a fast row count
    with open(input_file, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1 # Subtract header row
    
    chunk_size = math.ceil(total_rows / num_files)
    
    print(f"Total rows to process: {total_rows:,}")
    print(f"Targeting {num_files} files with ~{chunk_size:,} rows each.")

    # 3. Read and Write
    # 'chunksize' ensures we don't crash the RAM
    reader = pd.read_csv(
        input_file, 
        chunksize=chunk_size, 
        engine='python',
        quoting=1 # This handles the double quotes in your example data correctly
    )

    for i, chunk in enumerate(reader):
        if i >= num_files:
            # Append any leftover rows to the last file to ensure 100% data retention
            last_file = os.path.join(output_folder, f"ner_output_{num_files-1}.csv")
            chunk.to_csv(last_file, mode='a', index=False, header=False, quoting=1)
            continue
            
        output_filename = f"ner_output_{i}.csv"
        output_path = os.path.join(output_folder, output_filename)
        
        chunk.to_csv(output_path, index=False, quoting=1)
        print(f"✅ Created: {output_path} ({len(chunk):,} rows)")

    print("\nDone! 50 files are ready in the 'unique_ner' folder.")

if __name__ == "__main__":
    # Ensure this matches your actual filename
    split_csv("combined_unique_ner.csv", "unique_ner", num_files=50)