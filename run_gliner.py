# ── Section 0: Imports & Config ──────────────────────────────────────────────
import json, os, re
import argparse
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from gliner import GLiNER
import torch
from constants import *
import sys

"""
python3 run_gliner.py -s 0 -i splits -o gliner_outputs
"""

# Set up argparse for dynamic cmd inputs
parser = argparse.ArgumentParser(description="Run GLiNER NER extraction pipeline.")
parser.add_argument("-s", "--split_id", type=int, required=True, help="Split index (e.g., from SLURM array)")
parser.add_argument("-i", "--input_dir", type=str, default="splits", help="Input folder containing JSONL splits")
parser.add_argument("-o", "--output_dir", type=str, default="gliner_outputs", help="Output folder for results and progress")
args = parser.parse_args()

split_id = args.split_id
SPLITS_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

# Get sorted list of files
if not os.path.exists(SPLITS_DIR):
    print(f"Error: Input directory '{SPLITS_DIR}' not found.")
    sys.exit(1)

split_files = sorted([f for f in os.listdir(SPLITS_DIR) if f.endswith(".jsonl")])

if split_id >= len(split_files):
    print(f"Error: split_id {split_id} is out of bounds. Only {len(split_files)} files found in {SPLITS_DIR}.")
    sys.exit(1)

DATASET_PATH = os.path.join(SPLITS_DIR, split_files[split_id])

# Output paths
os.makedirs(OUTPUT_DIR, exist_ok=True)

NER_OUT_PATH   = os.path.join(OUTPUT_DIR, f"ner_output_{split_id}.jsonl")
PROGRESS_PATH  = os.path.join(OUTPUT_DIR, f"progress_{split_id}.json")
WARNINGS_PATH  = os.path.join(OUTPUT_DIR, f"warnings_{split_id}.json")
CSV_OUT_PATH   = os.path.join(OUTPUT_DIR, f"ner_output_{split_id}.csv")

# Initialize files if needed
for path in [NER_OUT_PATH, PROGRESS_PATH, WARNINGS_PATH]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump([], f)

# ✅ Initialize CSV with header (ONLY ONCE)
if not os.path.exists(CSV_OUT_PATH):
    pd.DataFrame(columns=["id", "text", "label", "start", "end", "score"]) \
        .to_csv(CSV_OUT_PATH, index=False)

print(f"Processing split {split_id}: {DATASET_PATH}")
print(f"Saving outputs to: {OUTPUT_DIR}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load GLiNER
print("Loading GLiNER …")
gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
gliner_model.to(DEVICE)


# ── Section 1: Chunking ──────────────────────────────────────────────────────
def split_into_chunks(text: str) -> list[tuple[str, int]]:
    tokenizer = gliner_model.data_processor.transformer_tokenizer

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    i = 0
    while i < len(tokens):
        end = min(i + GLINER_MAX_TOKENS, len(tokens))

        char_start = offsets[i][0]
        char_end   = offsets[end - 1][1]

        chunk_text = text[char_start:char_end]
        chunks.append((chunk_text, char_start))

        if end == len(tokens):
            break

        i += GLINER_MAX_TOKENS - GLINER_STRIDE_TOKENS

    return chunks


# ── Section 2: Post-processing ───────────────────────────────────────────────
def resolve_overlaps(entities: list[dict]) -> list[dict]:
    sorted_ents = sorted(entities, key=lambda e: e["end"] - e["start"], reverse=True)

    kept = []
    for cand in sorted_ents:
        dominated = any(
            (k["start"] <= cand["start"] and k["end"] >= cand["end"]
             and not (k["start"] == cand["start"] and k["end"] == cand["end"]))
            for k in kept
        )
        if not dominated:
            kept.append(cand)

    return kept


def run_gliner_on_text(text: str) -> list[dict]:
    chunks = split_into_chunks(text)
    all_ents = []

    for chunk_text, offset in chunks:
        preds = gliner_model.predict_entities(
            chunk_text, LABELS, threshold=GLINER_THRESHOLD
        )

        for ent in preds:
            all_ents.append({
                "text": ent["text"],
                "label": ent["label"],
                "start": ent["start"] + offset,
                "end": ent["end"] + offset,
                "score": round(ent["score"], 4),
            })

    # Deduplicate
    seen = set()
    unique = []
    for e in all_ents:
        key = (e["start"], e["end"], e["label"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return resolve_overlaps(unique)


# ── Section 3: Safe wrapper (warnings) ───────────────────────────────────────
def run_gliner_on_text_safe(text: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entities = run_gliner_on_text(text)

    trunc = any(
        issubclass(w.category, UserWarning)
        and "truncated" in str(w.message).lower()
        for w in caught
    )

    return entities, trunc


# ── Section 4: Progress tracking ─────────────────────────────────────────────
def load_progress():
    if Path(PROGRESS_PATH).exists():
        return set(json.load(open(PROGRESS_PATH)))
    return set()

def save_progress(done_ids):
    json.dump(list(done_ids), open(PROGRESS_PATH, "w"))

def load_warned_ids():
    if Path(WARNINGS_PATH).exists():
        return set(json.load(open(WARNINGS_PATH)))
    return set()

def save_warned_ids(ids):
    json.dump(list(ids), open(WARNINGS_PATH, "w"))


# ── Section 5: MAIN PIPELINE ─────────────────────────────────────────────────
def run_ner_pipeline(max_articles=None):
    done_ids = load_progress()
    warned_ids = load_warned_ids()

    print(f"Resuming: {len(done_ids)} done")
    print(f"Warned : {len(warned_ids)} skipped")

    with jsonlines.open(DATASET_PATH) as reader:
        total = sum(1 for _ in reader)

    processed = 0

    with jsonlines.open(DATASET_PATH) as reader, \
         jsonlines.open(NER_OUT_PATH, mode="a") as writer:

        for article in tqdm(reader, total=total, desc="NER"):
            art_id = article["id"]

            if art_id in done_ids:
                continue

            text = ((article.get("title") or "") + " " +
                    (article.get("content") or "")).strip()

            if not text:
                done_ids.add(art_id)
                continue

            entities, truncated = run_gliner_on_text_safe(text)

            if truncated:
                warned_ids.add(art_id)
                done_ids.add(art_id)
                continue

            # ✅ Save JSONL (original)
            writer.write({"id": art_id, "entities": entities})

            # ✅ SAVE CSV IMMEDIATELY (PER ARTICLE)
            rows = [
                {
                    "id": art_id,
                    "text": ent["text"],
                    "label": ent["label"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "score": ent["score"],
                }
                for ent in entities
            ]

            if rows:
                with open(CSV_OUT_PATH, "a") as f:
                    pd.DataFrame(rows).to_csv(
                        f, header=False, index=False
                    )
                    f.flush()
                    os.fsync(f.fileno())

            done_ids.add(art_id)
            processed += 1

            if processed % 100 == 0:
                save_progress(done_ids)
                save_warned_ids(warned_ids)
                print(f"[{processed}] saved")

            if max_articles and processed >= max_articles:
                break

    save_progress(done_ids)
    save_warned_ids(warned_ids)

    print(f"\nDone. Processed: {processed}")


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ner_pipeline()