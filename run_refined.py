import sys
import json
import os
from pathlib import Path

import pandas as pd
import jsonlines
from tqdm import tqdm

# Import ReFinED
from refined.inference.processor import Refined

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python perform_el.py <task_id>")
    sys.exit(1)

TASK_ID = int(sys.argv[1])

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR = "unique_ner_gliner_final"
NER_CSV_PATH = os.path.join(INPUT_DIR, f"ner_output_{TASK_ID}.csv")

OUTPUT_DIR = "unique_el_final"
PROGRESS_DIR = "unique_el_progress_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

EL_OUT_PATH      = os.path.join(OUTPUT_DIR, f"el_output_refined_{TASK_ID}.jsonl")
EL_CSV_PATH      = os.path.join(OUTPUT_DIR, f"el_output_refined_{TASK_ID}.csv")
EL_PROGRESS_PATH = os.path.join(PROGRESS_DIR, f"el_progress_refined_{TASK_ID}.json")

# ── Load ReFinED ──────────────────────────────────────────────────────────────
print(f"[Task {TASK_ID}] Loading ReFinED (Local Database)...")
refined = Refined.from_pretrained(
    model_name='wikipedia_model_with_numbers',
    entity_set="wikipedia"
)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(NER_CSV_PATH):
        print(f"File not found: {NER_CSV_PATH}")
        return

    # Load NER data - Processing ALL labels now
    df = pd.read_csv(NER_CSV_PATH, engine="c", on_bad_lines="warn", dtype=str)
    df = df.dropna(subset=["id", "text", "label", "start", "end", "score"])
    
    # Load progress
    done_keys = set()
    if Path(EL_PROGRESS_PATH).exists():
        with open(EL_PROGRESS_PATH) as f:
            done_keys = set(json.load(f))

    # Progress key
    df["_el_key"] = df["id"].astype(str) + "||" + df["start"].astype(str)
    pending_df = df[~df["_el_key"].isin(done_keys)].reset_index(drop=True)

    if pending_df.empty:
        print(f"[Task {TASK_ID}] Nothing to do.")
        return

    with jsonlines.open(EL_OUT_PATH, mode="a") as writer:
        
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
            
            # Save the record (using your updated column names)
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
                with open(EL_PROGRESS_PATH, "w") as f:
                    json.dump(list(done_keys), f)

        # Final progress save
        with open(EL_PROGRESS_PATH, "w") as f:
            json.dump(list(done_keys), f)

    # Export to CSV
    if os.path.exists(EL_OUT_PATH):
        with jsonlines.open(EL_OUT_PATH) as reader:
            all_results = list(reader)
        if all_results:
            df_csv = pd.DataFrame(all_results)
            # Cleanly export the CSV without trying to look up removed keys
            df_csv.to_csv(EL_CSV_PATH, index=False)
            print(f"\n[Task {TASK_ID}] ✅ Complete. All entities processed.")

if __name__ == "__main__":
    main()