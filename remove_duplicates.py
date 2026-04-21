import os
import pandas as pd
import argparse

def remove_duplicates_from_csv(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
    
    original_count = len(df)
    print(f"Total rows loaded: {original_count}")
    
    # 1. Clean the 'text' column (Remove newlines and extra spaces)
    print("Cleaning text formatting...")
    # Convert to string just in case there are completely numeric entity names
    df['text'] = df['text'].astype(str) 
    df['text'] = df['text'].replace(r'\n', ' ', regex=True)
    df['text'] = df['text'].replace(r'\s+', ' ', regex=True).str.strip()

    # 2. Remove duplicates (Keep the one with the highest score)
    print("Removing duplicates...")
    # Sort by score descending first so the highest score is at the top
    df = df.sort_values(by='score', ascending=False)
    # Drop duplicates matching the exact same text and label
    df = df.drop_duplicates(subset=['text', 'label'], keep='first')
    
    unique_count = len(df)
    
    # 3. Enforce column order and save to CSV
    df = df[['id', 'text', 'label', 'start', 'end', 'score']]
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 4. Print summary
    print("\n--- Summary ---")
    print(f"Original entities:      {original_count}")
    print(f"Duplicates removed:     {original_count - unique_count}")
    print(f"Final unique entities:  {unique_count}")
    print(f"Success! Cleaned data saved to: {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Remove duplicates from NER output CSV.")
    
    # Add arguments for input and output files
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="The path to the input CSV file (e.g., outputs_gliner/combined_ner_output.csv)."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, 
        help="The path to save the cleaned output CSV file (e.g., outputs_gliner/unique_ner_output.csv)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the function with the parsed arguments
    remove_duplicates_from_csv(args.input, args.output)