# ── Section 0: Imports & Config ──────────────────────────────────────────────
import json, os, re
from pathlib import Path

import jsonlines
import pandas as pd
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from constants import *
import sys

# Get split index from SLURM
split_id = int(sys.argv[1])

# Input splits folder
SPLITS_DIR = "splits"

# Get sorted list of files
split_files = sorted([f for f in os.listdir(SPLITS_DIR) if f.endswith(".jsonl")])

DATASET_PATH = os.path.join(SPLITS_DIR, split_files[split_id])

# Make output unique per split
NER_OUT_PATH = f"outputs/ner_output_{split_id}.jsonl"
PROGRESS_PATH = f"outputs/progress_{split_id}.json"
WARNINGS_PATH = f"outputs/warnings_{split_id}.json"

print(f"Processing split {split_id}: {DATASET_PATH}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# GLiNER
print("Loading GLiNER …")
gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
gliner_model.to(DEVICE)

# ── Section 2: GLiNER helpers ─────────────────────────────────────────────────

# ── Section 2: replace split_into_chunks with a token-aware version ───────────

def split_into_chunks(text: str) -> list[tuple[str, int]]:
    """
    Splits *text* into overlapping chunks that each stay within
    GLINER_MAX_TOKENS subword tokens — using GLiNER's own tokenizer
    so the count is exact and no truncation warnings fire.

    Returns list of (chunk_text, char_offset_in_original).
    """
    tokenizer = gliner_model.data_processor.transformer_tokenizer

    # Encode the whole document once to get the token↔char mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,   # GLiNER adds its own specials
        truncation=False,
    )

    tokens         = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]   # (char_start, char_end) per token
    total_tokens   = len(tokens)

    chunks = []
    i = 0
    while i < total_tokens:
        end = min(i + GLINER_MAX_TOKENS, total_tokens)

        # Char span that covers tokens[i:end]
        char_start = offset_mapping[i][0]
        char_end   = offset_mapping[end - 1][1]
        chunk_text = text[char_start:char_end]

        chunks.append((chunk_text, char_start))

        if end == total_tokens:
            break
        i += GLINER_MAX_TOKENS - GLINER_STRIDE_TOKENS   # slide forward with stride

    return chunks


def resolve_overlaps(entities: list[dict]) -> list[dict]:
    """
    When entity A's span is fully contained within entity B's span,
    keep B (the longer / more specific mention) and discard A.
    Example: 'Alberta' inside 'University of Alberta' → keep the latter.
    """
    # Sort by span length descending so longer spans are checked first
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
    """
    Runs GLiNER on *text* with chunking, remaps offsets to the full document,
    resolves overlapping spans, and returns a deduplicated entity list.
    Each entity dict: {text, label, start, end, score}
    """
    chunks   = split_into_chunks(text)
    all_ents = []

    for chunk_text, char_offset in chunks:
        preds = gliner_model.predict_entities(
            chunk_text, LABELS, threshold=GLINER_THRESHOLD
        )
        for ent in preds:
            all_ents.append({
                "text":  ent["text"],
                "label": ent["label"],
                "start": ent["start"] + char_offset,   # remap to doc offset
                "end":   ent["end"]   + char_offset,
                "score": round(ent["score"], 4),
            })

    # Remove span-level duplicates from chunk overlaps
    seen = set()
    unique_ents = []
    for e in all_ents:
        key = (e["start"], e["end"], e["label"])
        if key not in seen:
            seen.add(key)
            unique_ents.append(e)

    return resolve_overlaps(unique_ents)

# ── Section 3: NER with progress tracking + warning capture ──────────────────
def load_progress() -> set:
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()

def save_progress(done_ids: set):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done_ids), f)

def load_warned_ids() -> set:
    if Path(WARNINGS_PATH).exists():
        with open(WARNINGS_PATH) as f:
            return set(json.load(f))
    return set()

def save_warned_ids(warned_ids: set):
    with open(WARNINGS_PATH, "w") as f:
        json.dump(list(warned_ids), f)


def run_gliner_on_text_safe(text: str) -> tuple[list[dict], bool]:
    """
    Wraps run_gliner_on_text with warning capture.
    Returns (entities, was_truncated).
    was_truncated=True means a truncation warning fired during processing.
    """
    triggered = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entities = run_gliner_on_text(text)

    truncation_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "truncated" in str(w.message).lower()
    ]

    return entities, len(truncation_warnings) > 0


def run_ner_pipeline(max_articles: int | None = None):
    """
    Streams through the dataset, skips already-processed IDs,
    writes NER results line-by-line to NER_OUT_PATH.
    Articles that trigger GLiNER truncation warnings are logged to
    WARNINGS_PATH and excluded from the results file.
    Safe to interrupt and resume at any time.
    """
    done_ids   = load_progress()
    warned_ids = load_warned_ids()
    print(f"Resuming  : {len(done_ids)} articles already processed.")
    print(f"Warned IDs: {len(warned_ids)} articles previously flagged & skipped.")

    with jsonlines.open(DATASET_PATH) as reader:
        total = sum(1 for _ in reader)

    processed    = 0
    new_warnings = 0

    with jsonlines.open(DATASET_PATH) as reader, \
         jsonlines.open(NER_OUT_PATH, mode="a") as writer:

        for article in tqdm(reader, total=total, desc="NER"):
            art_id = article["id"]

            if art_id in done_ids:
                continue

            full_text = (article.get("title") or "") + " " + \
                        (article.get("content") or "")
            full_text = full_text.strip()

            if not full_text:
                done_ids.add(art_id)
                continue

            entities, was_truncated = run_gliner_on_text_safe(full_text)

            if was_truncated:
                # Log it, mark as done so we don't retry it, but don't write results
                warned_ids.add(art_id)
                done_ids.add(art_id)
                new_warnings += 1
            else:
                writer.write({"id": art_id, "entities": entities})
                done_ids.add(art_id)

            processed += 1

            if processed % 100 == 0:
                save_progress(done_ids)
                save_warned_ids(warned_ids)
                print(f"  [{processed}] warnings this run: {new_warnings}")

            if max_articles and processed >= max_articles:
                print(f"Stopped early after {processed} articles.")
                break

    save_progress(done_ids)
    save_warned_ids(warned_ids)
    print(f"\nDone. Processed: {processed} | Truncation-warned (excluded): {new_warnings}")
    print(f"Warning IDs saved to: {WARNINGS_PATH}")


run_ner_pipeline(max_articles=None)