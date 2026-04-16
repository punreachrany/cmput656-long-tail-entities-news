# ── Section 0: Imports & Config ──────────────────────────────────────────────
import json
import os
import sys
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm

from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger

# Get split index from SLURM
split_id = int(sys.argv[1])

# Input splits folder
SPLITS_DIR = "splits"

# Get sorted list of files
split_files = sorted([f for f in os.listdir(SPLITS_DIR) if f.endswith(".jsonl")])
DATASET_PATH = os.path.join(SPLITS_DIR, split_files[split_id])

# Output paths
NER_OUT_PATH   = f"outputs/ner_output_{split_id}.jsonl"
PROGRESS_PATH  = f"outputs/progress_{split_id}.json"
WARNINGS_PATH  = f"outputs/warnings_{split_id}.json"
CSV_OUT_PATH   = f"outputs/ner_output_{split_id}.csv"

# Create output dir
os.makedirs("outputs", exist_ok=True)

# Initialize files if needed
for path in [NER_OUT_PATH, PROGRESS_PATH, WARNINGS_PATH]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump([], f)

# Initialize CSV with header (ONLY ONCE)
if not os.path.exists(CSV_OUT_PATH):
    pd.DataFrame(columns=["id", "text", "label", "start", "end", "score"]) \
        .to_csv(CSV_OUT_PATH, index=False)

print(f"Processing split {split_id}: {DATASET_PATH}")

# ── Setup Stanford NER ───────────────────────────────────────────────────────
# IMPORTANT: Update these paths to point to where you unzipped Stanford NER
PATH_TO_JAR = 'stanford-ner.jar'
PATH_TO_MODEL = 'classifiers/english.all.3class.distsim.crf.ser.gz'

print("Loading Stanford NER …")
try:
    st = StanfordNERTagger(PATH_TO_MODEL, PATH_TO_JAR, encoding='utf-8')
    tokenizer = TreebankWordTokenizer()
except Exception as e:
    print(f"Failed to load Stanford NER. Ensure Java is installed and paths are correct. Error: {e}")
    sys.exit(1)


# ── Section 1: NER Extraction ────────────────────────────────────────────────
def run_stanford_on_text(text: str) -> list[dict]:
    if not text.strip():
        return []

    # Get tokens and their exact character spans in the original text
    spans = list(tokenizer.span_tokenize(text))
    tokens = [text[start:end] for start, end in spans]

    if not tokens:
        return []

    # Run Stanford NER
    tagged_tokens = st.tag(tokens)

    entities = []
    current_ent = None

    # Group contiguous tokens with the same NER tag into a single entity
    for (token, tag), (start, end) in zip(tagged_tokens, spans):
        if tag != 'O':
            # If the current token shares the same tag as the previous one and is adjacent
            if current_ent and current_ent['label'] == tag and (start - current_ent['end'] <= 2):
                current_ent['end'] = end
                current_ent['text'] = text[current_ent['start']:end]
            else:
                if current_ent:
                    entities.append(current_ent)
                current_ent = {
                    "text": text[start:end],
                    "label": tag,
                    "start": start,
                    "end": end,
                    "score": 1.0  # Stanford NER Python wrapper does not provide confidence scores
                }
        else:
            if current_ent:
                entities.append(current_ent)
                current_ent = None
                
    if current_ent:
        entities.append(current_ent)

    return entities


# ── Section 2: Progress tracking ─────────────────────────────────────────────
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


# ── Section 3: MAIN PIPELINE ─────────────────────────────────────────────────
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

            # Run Stanford NER
            entities = run_stanford_on_text(text)

            # Save JSONL (original)
            writer.write({"id": art_id, "entities": entities})

            # SAVE CSV IMMEDIATELY (PER ARTICLE)
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