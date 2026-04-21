import os

# --- CONFIGURATION ---
INPUT_FILE = "sample-1M.jsonl"
OUTPUT_DIR = "splits"
NUM_SPLITS = 10

def split_jsonl():
    # 1. Create the splits directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Count total lines in the input file
    print(f"Counting lines in '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    print(f"Total lines found: {total_lines}")

    # 3. Calculate how many lines each file should get
    base_lines = total_lines // NUM_SPLITS
    remainder = total_lines % NUM_SPLITS

    # 4. Read the input file again and write to the output files
    print(f"Splitting into {NUM_SPLITS} files...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        for i in range(1, NUM_SPLITS + 1):
            output_path = os.path.join(OUTPUT_DIR, f"part_{i}.jsonl")
            
            # Distribute the remainder by adding 1 extra line to the first few files
            current_chunk_size = base_lines + (1 if i <= remainder else 0)
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for _ in range(current_chunk_size):
                    line = f_in.readline()
                    if not line:
                        break # Safety break if we hit end of file early
                    f_out.write(line)
            
            print(f"Created {output_path} ({current_chunk_size} lines)")

    print("\n✅ Done! All files have been successfully split.")

if __name__ == "__main__":
    split_jsonl()