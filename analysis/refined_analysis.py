import os
import pandas as pd
import math

def split_csv_into_chunks(input_file, output_folder, num_splits=100):
    # 1. Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
        
    total_rows = len(df)
    print(f"Total rows loaded: {total_rows}")
    print(f"Splitting into {num_splits} files...")

    # 2. Calculate the maximum number of rows per file
    chunk_size = math.ceil(total_rows / num_splits)
    
    # 3. Slice the dataframe and save each chunk
    for i in range(num_splits):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, total_rows)
        
        # Use .iloc to slice the dataframe safely
        chunk = df.iloc[start_row:end_row]
        
        # Only save if the chunk isn't empty
        if not chunk.empty:
            output_filename = f"ner_output_{i}.csv"
            output_path = os.path.join(output_folder, output_filename)
            
            chunk.to_csv(output_path, index=False, encoding='utf-8')
            
    print(f"Success! Generated files inside the '{output_folder}' directory.")
    print(f"Max rows per file: {chunk_size}")

# --- Run the script ---
if __name__ == "__main__":
    INPUT_FILE = "outputs_gliner/unique_ner_output.csv"
    OUTPUT_FOLDER = "unique_ner_gliner_final"
    
    split_csv_into_chunks(INPUT_FILE, OUTPUT_FOLDER, num_splits=1000)