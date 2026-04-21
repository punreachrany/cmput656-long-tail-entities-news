import sys
import pandas as pd
import glob
import os

def process_and_deduplicate(task_type):
    if task_type == "ner":
        input_folder = "ner_outputs"
        file_pattern = "ner_output_*.csv"
        output_csv = "combined_unique_ner.csv"
        text_col = "text"
        label_col = "label"
        report_filename = "report_ner.txt"
    elif task_type == "el":
        input_folder = "el_outputs"
        file_pattern = "el_output_*.csv"
        output_csv = "combined_unique_el.csv"
        text_col = "text"
        label_col = "label"
        report_filename = "report_el.txt"
    else:
        print("Invalid task type. Please use 'ner' or 'el'.")
        return

    print(f"[{task_type.upper()}] Starting process...")
    search_path = os.path.join(input_folder, file_pattern)
    all_files = sorted(glob.glob(search_path))

    if not all_files:
        print(f"[{task_type.upper()}] ⚠️ No files found for pattern: {search_path}")
        return

    # 1. Load data
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, engine="python", on_bad_lines="skip")
            df_list.append(df)
        except Exception as e:
            print(f"[{task_type.upper()}] ❌ Error loading {file}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    initial_count = len(combined_df)
    
    # 2. Normalize and Deduplicate
    combined_df["_temp_norm_text"] = combined_df[text_col].astype(str).str.strip().str.lower()

    if "score" in combined_df.columns:
        combined_df["score"] = pd.to_numeric(combined_df["score"], errors="coerce")
        combined_df = combined_df.sort_values(by="score", ascending=False)

    df_unique = combined_df.drop_duplicates(subset=["_temp_norm_text", label_col], keep="first")
    df_unique = df_unique.drop(columns=["_temp_norm_text"])

    final_count = len(df_unique)
    deleted_count = initial_count - final_count

    # 3. Save to CSV
    df_unique.to_csv(output_csv, index=False, encoding="utf-8", quoting=1)
    print(f"[{task_type.upper()}] ✅ Saved {final_count} unique rows to {output_csv}")

    # 4. Write temporary report
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"--- {task_type.upper()} DATASET ---\n")
        f.write(f"Total rows before deduplication : {initial_count:,}\n")
        f.write(f"Total duplicate rows deleted    : {deleted_count:,}\n")
        f.write(f"Total unique rows remaining     : {final_count:,}\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parallel_dedup.py [ner|el]")
        sys.exit(1)

    task = sys.argv[1].lower()
    process_and_deduplicate(task)