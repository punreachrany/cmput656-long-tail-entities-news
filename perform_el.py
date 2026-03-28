"""
perform_el.py <task_id>

Reads  : outputs/ner_output_<task_id>.csv
Writes : el_outputs/el_output_<task_id>.jsonl
Tracks : el_progress/el_progress_<task_id>.json

Resumable — safe to cancel and resubmit.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python perform_el.py <task_id>")
    sys.exit(1)

TASK_ID = int(sys.argv[1])

# ── Paths ─────────────────────────────────────────────────────────────────────
NER_CSV_PATH     = f"outputs/ner_output_{TASK_ID}.csv"
EL_OUT_PATH      = f"el_outputs/el_output_{TASK_ID}.jsonl"
EL_CSV_PATH      = f"el_outputs/el_output_{TASK_ID}.csv"        # ← NEW
EL_PROGRESS_PATH = f"el_progress/el_progress_{TASK_ID}.json"

# ── Config ────────────────────────────────────────────────────────────────────
GENRE_MODEL_NAME = "facebook/genre-linking-aidayago2"
GENRE_BATCH_SIZE = 32   # H100 can handle larger batches
GENRE_NUM_BEAMS  = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Task {TASK_ID}] Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[Task {TASK_ID}] Loading GENRE …")
genre_tokenizer = AutoTokenizer.from_pretrained(GENRE_MODEL_NAME)
genre_model     = AutoModelForSeq2SeqLM.from_pretrained(GENRE_MODEL_NAME)
genre_model.to(DEVICE)
genre_model.eval()
print(f"[Task {TASK_ID}] Model ready.")

# ── CSV loader ────────────────────────────────────────────────────────────────

def load_ner_csv(path: str) -> pd.DataFrame:
    """
    Reads NER output CSV safely.
    Loads all columns as strings first, then casts and drops malformed rows.
    """
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="warn",
        dtype=str,   # load everything as string — no casting errors
    )

    print(f"[Task {TASK_ID}] Raw rows loaded: {len(df):,}")

    # Drop rows where any critical field is missing
    df = df.dropna(subset=["id", "text", "label", "start", "end", "score"])

    # Now safe to cast
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    df["score"] = df["score"].astype(float)

    # Drop rows that look like garbage (artifact of split newlines)
    # A valid row must have a non-empty text and a known label
    valid_labels = {
        "person", "norp", "facility", "organization", "gpe", "location",
        "product", "event", "work_of_art", "law", "language",
        "date", "time", "percent", "money", "quantity", "ordinal", "cardinal",
        "religion", "political_party", "nationality", "ethnic_group",
        "title", "award", "disease", "chemical", "weapon",
        "vehicle", "currency", "brand"
    }
    before = len(df)
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    dropped = before - len(df)

    if dropped:
        print(f"[Task {TASK_ID}] Dropped {dropped:,} malformed rows.")

    print(f"[Task {TASK_ID}] Clean rows: {len(df):,} from {path}")
    return df

# ── GENRE helpers ─────────────────────────────────────────────────────────────

def build_genre_input(surface: str, context: str = "") -> str:
    if context:
        idx = context.lower().find(surface.lower())
        if idx != -1:
            prefix = context[max(0, idx - 100): idx]
            suffix = context[idx + len(surface): idx + len(surface) + 100]
            return f"{prefix} [START_ENT] {surface} [END_ENT] {suffix}"
    return f"[START_ENT] {surface} [END_ENT]"


def genre_link_batch(surfaces: list[str],
                     contexts: list[str] | None = None) -> list[str]:
    if contexts is None:
        contexts = [""] * len(surfaces)

    inputs_text = [build_genre_input(s, c) for s, c in zip(surfaces, contexts)]
    inputs = genre_tokenizer(
        inputs_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = genre_model.generate(
            **inputs,
            num_beams=GENRE_NUM_BEAMS,
            num_return_sequences=1,
            max_length=64,
        )
    return genre_tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ── Progress helpers ──────────────────────────────────────────────────────────

def el_progress_key(norm_text: str, label: str) -> str:
    return f"{norm_text}||{label}"


def load_el_progress() -> set:
    if Path(EL_PROGRESS_PATH).exists():
        with open(EL_PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def save_el_progress(done_keys: set):
    with open(EL_PROGRESS_PATH, "w") as f:
        json.dump(list(done_keys), f)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load NER CSV for this split
    df = load_ner_csv(NER_CSV_PATH)

    # Rename 'id' → 'article_id' to be explicit
    df = df.rename(columns={"id": "article_id"})

    # Add norm_text for dedup tracking
    df["norm_text"] = df["text"].str.strip().str.lower()

    # Resume: skip already-linked keys
    done_keys = load_el_progress()
    print(f"[Task {TASK_ID}] Already linked : {len(done_keys):,}")

    df["_el_key"] = df.apply(
        lambda r: el_progress_key(r["norm_text"], r["label"]), axis=1
    )

    # Deduplicate on (norm_text, label) — no need to link the same entity twice
    df_dedup = (
        df.sort_values("score", ascending=False)
          .drop_duplicates(subset=["norm_text", "label"])
          .reset_index(drop=True)
    )
    print(f"[Task {TASK_ID}] Unique (text, label) : {len(df_dedup):,}")

    pending_df = df_dedup[~df_dedup["_el_key"].isin(done_keys)].reset_index(drop=True)
    print(f"[Task {TASK_ID}] Pending             : {len(pending_df):,}")

    if pending_df.empty:
        print(f"[Task {TASK_ID}] Nothing to do — exiting.")
        return

    new_linked = 0

    with jsonlines.open(EL_OUT_PATH, mode="a") as writer:
        for i in tqdm(
            range(0, len(pending_df), GENRE_BATCH_SIZE),
            desc=f"EL split {TASK_ID}",
        ):
            batch_df = pending_df.iloc[i: i + GENRE_BATCH_SIZE]
            surfaces = batch_df["text"].tolist()
            preds    = genre_link_batch(surfaces)

            for (_, row), wiki_title in zip(batch_df.iterrows(), preds):
                writer.write({
                    "article_id": row["article_id"],
                    "text":       row["text"],
                    "norm_text":  row["norm_text"],
                    "label":      row["label"],
                    "score":      row["score"],
                    "wiki_title": wiki_title,
                    "split":      TASK_ID,
                })
                done_keys.add(row["_el_key"])
                new_linked += 1

            # Flush progress after every batch
            save_el_progress(done_keys)

    print(f"[Task {TASK_ID}] Done. Linked this run : {new_linked:,}")
    print(f"[Task {TASK_ID}]   JSONL    → {EL_OUT_PATH}")
    print(f"[Task {TASK_ID}]   Progress → {EL_PROGRESS_PATH}")

    # ── Save CSV snapshot ─────────────────────────────────────────────────────
    # Read the full JSONL (includes results from previous resumed runs)
    with jsonlines.open(EL_OUT_PATH) as reader:
        all_results = list(reader)

    if all_results:
        el_df = pd.DataFrame(all_results)
        el_df.to_csv(
            EL_CSV_PATH,
            index=False,
            quoting=1,       # QUOTE_ALL — safe for text fields with commas
            encoding="utf-8",
        )
        print(f"[Task {TASK_ID}]   CSV      → {EL_CSV_PATH} ({len(el_df):,} rows total)")
    else:
        print(f"[Task {TASK_ID}]   CSV skipped — no results to write.")


if __name__ == "__main__":
    main()