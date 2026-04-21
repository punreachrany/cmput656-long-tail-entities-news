import os
import argparse
"""
python3 split_data.py -i sample-1M.jsonl -o splits
python3 split_data.py -i gliner_outputs/unique_gliner_output.csv -o unique_ner
python3 split_data.py -i sample-1M.jsonl -o splits -n 5
"""

def split_file(input_file, output_dir, num_splits):
    # 1. Create the splits directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the file extension (e.g., '.jsonl' or '.csv')
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()

    # 2. Count total lines in the input file
    print(f"Counting lines in '{input_file}'...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    print(f"Total lines found: {total_lines}")

    # If it's a CSV, we need to account for the header row
    is_csv = (ext == '.csv')
    data_lines = total_lines - 1 if is_csv and total_lines > 0 else total_lines

    if data_lines <= 0:
        print("Error: Not enough data lines to split.")
        return

    # 3. Calculate how many lines each file should get
    base_lines = data_lines // num_splits
    remainder = data_lines % num_splits

    # 4. Read the input file again and write to the output files
    print(f"Splitting into {num_splits} files...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        # Read the header first if it's a CSV
        header = f_in.readline() if is_csv else ""

        for i in range(1, num_splits + 1):
            # Use the extracted extension dynamically
            output_path = os.path.join(output_dir, f"part_{i}{ext}")
            
            # Distribute the remainder by adding 1 extra line to the first few files
            current_chunk_size = base_lines + (1 if i <= remainder else 0)
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Write the header at the top of every new CSV file
                if is_csv:
                    f_out.write(header)

                for _ in range(current_chunk_size):
                    line = f_in.readline()
                    if not line:
                        break # Safety break if we hit end of file early
                    f_out.write(line)
            
            print(f"Created {output_path} ({current_chunk_size} data lines)")

    print("\n✅ Done! All files have been successfully split.")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split a file (JSONL or CSV) into multiple smaller files.")
    
    # Add arguments for input file, output directory, and optionally number of splits
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the input file (e.g., data.jsonl or data.csv)."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the split files (e.g., splits)."
    )
    parser.add_argument(
        "-n", "--num_splits", 
        type=int, 
        default=10, 
        help="Number of files to split into (default: 10)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the function with the parsed arguments
    split_file(args.input, args.output_dir, args.num_splits)