import os
import json
import argparse
from pathlib import Path

import pandas as pd
import jsonlines
from tqdm import tqdm

"""
python3 run_refined.py -t 1 -i unique_ner -o el_outputs
"""

# Import ReFinED
from refined.inference.processor import Refined

def main(task_id, input_dir, output_dir):
    # ── Paths ─────────────────────────────────────────────────────────────────
    ner_csv_path = os.path.join(input_dir, f"part_{task_id}.csv")

    # Dynamically create the progress directory based on the output directory name
    progress_dir = f"{output_dir}_progress"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    el_out_path      = os.path.join(output_dir, f"el_output_{task_id}.jsonl")
    el_csv_path      = os.path.join(output_dir, f"el_output_{task_id}.csv")
    el_progress_path = os.path.join(progress_dir, f"el_progress_{task_id}.json")

    # ── Check Input ───────────────────────────────────────────────────────────
    if not os.path.exists(ner_csv_path):
        print(f"Error: File not found: {ner_csv_path}")
        return

    # ── Load ReFinED ──────────────────────────────────────────────────────────
    print(f"[Task {task_id}] Loading ReFinED (Local Database)...")
    refined = Refined.from_pretrained(
        model_name='wikipedia_model_with_numbers',
        entity_set="wikipedia"
    )

    # ── Processing ────────────────────────────────────────────────────────────
    # Load NER data - Processing ALL labels now
    df = pd.read_csv(ner_csv_path, engine="c", on_bad_lines="warn", dtype=str)
    df = df.dropna(subset=["id", "text", "label", "start", "end", "score"])
    
    # Load progress
    done_keys = set()
    if Path(el_progress_path).exists():
        with open(el_progress_path) as f:
            done_keys = set(json.load(f))

    # Progress key
    df["_el_key"] = df["id"].astype(str) + "||" + df["start"].astype(str)
    pending_df = df[~df["_el_key"].isin(done_keys)].reset_index(drop=True)

    if pending_df.empty:
        print(f"[Task {task_id}] Nothing to do. All entities already processed.")
        return

    with jsonlines.open(el_out_path, mode="a") as writer:
        
        for idx, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Linking"):
            mention_text = str(row["text"])
            
            # ReFinED inference
            spans = refined.process_text(mention_text)
            
            el_id = None
            el_label = None
            el_coarse_type = None
            
            if spans and len(spans) > 0:
                span = spans[0]
                if span.predicted_entity:
                    el_id = span.predicted_entity.wikidata_entity_id
                    el_label = span.predicted_entity.wikipedia_entity_title
                el_coarse_type = span.coarse_mention_type
            
            # Save the record
            writer.write({
                "ner_id": row["id"], 
                "el_id": el_id,
                "ner_text": mention_text, 
                "ner_type": row["label"],
                "el_text": el_label, 
                "el_type": el_coarse_type,
                "ner_score": float(row["score"])
            })
            done_keys.add(row["_el_key"])
            
            if idx % 100 == 0:
                with open(el_progress_path, "w") as f:
                    json.dump(list(done_keys), f)

        # Final progress save
        with open(el_progress_path, "w") as f:
            json.dump(list(done_keys), f)

    # Export to CSV
    if os.path.exists(el_out_path):
        with jsonlines.open(el_out_path) as reader:
            all_results = list(reader)
        if all_results:
            df_csv = pd.DataFrame(all_results)
            # Cleanly export the CSV
            df_csv.to_csv(el_csv_path, index=False)
            print(f"\n[Task {task_id}] ✅ Complete. All entities processed.")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run ReFinED Entity Linking on NER outputs.")
    
    # Add arguments for task ID, input directory, and output directory
    parser.add_argument(
        "-t", "--task_id", 
        type=int, 
        required=True, 
        help="The numerical task ID (e.g., 1, 2, 3)."
    )
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        default="unique_ner", 
        help="Directory containing the input NER CSV files (default: unique_ner)."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default="unique_el_final", 
        help="Directory to save the EL outputs (default: unique_el_final)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function
    main(args.task_id, args.input_dir, args.output_dir)