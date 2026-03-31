import sys
import json
import os
import time
from pathlib import Path

import pandas as pd
import jsonlines
import torch
from tqdm import tqdm
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python perform_el.py <task_id>")
    sys.exit(1)

TASK_ID = int(sys.argv[1])

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR = "unique_ner"
NER_CSV_PATH = os.path.join(INPUT_DIR, f"ner_output_{TASK_ID}.csv")

OUTPUT_DIR = "unique_el"
# Match the .sh script directory name
PROGRESS_DIR = "unique_el_progress" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

EL_OUT_PATH      = os.path.join(OUTPUT_DIR, f"el_output_{TASK_ID}.jsonl")
EL_CSV_PATH      = os.path.join(OUTPUT_DIR, f"el_output_{TASK_ID}.csv")
EL_PROGRESS_PATH = os.path.join(PROGRESS_DIR, f"el_progress_{TASK_ID}.json")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/genre-linking-blink"
BATCH_SIZE = 64  # Increased for H100 efficiency

VALID_LABELS = {
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language",
    "date", "time", "percent", "money", "quantity", "ordinal", "cardinal",
    "religion", "political_party", "nationality", "ethnic_group",
    "title", "award", "disease", "chemical", "weapon",
    "vehicle", "currency", "brand",
}

# Update this line in perform_el.py
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" # MacBook GPU acceleration
else:
    DEVICE = "cpu"
print(f"[Task {TASK_ID}] Device: {DEVICE}")

# ── Load GENRE ────────────────────────────────────────────────────────────────
print(f"[Task {TASK_ID}] Loading GENRE...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ── Wikidata helpers ──────────────────────────────────────────────────────────
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPARQL_API = "https://query.wikidata.org/sparql"
wiki_cache = {}
type_cache = {}
# CRITICAL: Use a specific User-Agent to avoid blocks
HEADERS = {"User-Agent": "ResearchEntityLinkingBot/1.0 (rany@ualberta.ca)"}

def get_qid(title):
    if not title: return None
    if title in wiki_cache: return wiki_cache[title]
    
    params = {"action": "query", "prop": "pageprops", "titles": title, "format": "json"}
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=5).json()
        pages = r.get("query", {}).get("pages", {})
        for p in pages.values():
            qid = p.get("pageprops", {}).get("wikibase_item")
            if qid:
                wiki_cache[title] = qid
                return qid
    except: pass
    return None

def get_types(qid):
    if not qid: return []
    if qid in type_cache: return type_cache[qid]
    
    # LIMIT 1 added to speed up requests since you only use the first element
    query = f"""
    SELECT ?typeLabel WHERE {{ 
        wd:{qid} wdt:P31 ?type. 
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} 
    }} LIMIT 1
    """
    try:
        r = requests.get(SPARQL_API, params={"query": query, "format": "json"}, headers=HEADERS, timeout=10).json()
        types = [x["typeLabel"]["value"] for x in r["results"]["bindings"]]
        type_cache[qid] = types
        time.sleep(0.1) # Reduced slightly due to LIMIT 1
        return types
    except: return []

def map_coarse(types):
    if not types: return None
    t = " ".join(types).lower()
    if "human" in t: return "PER"
    if any(x in t for x in ["company", "organization", "enterprise", "business"]): return "ORG"
    if any(x in t for x in ["city", "country", "state", "continent"]): return "LOC"
    return "MISC"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(NER_CSV_PATH):
        print(f"File not found: {NER_CSV_PATH}")
        return

    # Loading with specific engine for speed
    df = pd.read_csv(NER_CSV_PATH, engine="c", on_bad_lines="warn", dtype=str)
    df = df.dropna(subset=["id", "text", "label", "start", "end", "score"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    
    done_keys = set()
    if Path(EL_PROGRESS_PATH).exists():
        with open(EL_PROGRESS_PATH) as f:
            done_keys = set(json.load(f))

    # Progress key includes text to prevent collisions if IDs aren't unique
    df["_el_key"] = df["id"].astype(str) + "||" + df["start"].astype(str)
    pending_df = df[~df["_el_key"].isin(done_keys)].reset_index(drop=True)

    if pending_df.empty:
        print(f"[Task {TASK_ID}] Nothing to do.")
        return

    with jsonlines.open(EL_OUT_PATH, mode="a") as writer:
        # Step 1: Model Inference (Fast on GPU)
        for i in tqdm(range(0, len(pending_df), BATCH_SIZE), desc="Linking"):
            batch_df = pending_df.iloc[i : i+BATCH_SIZE]
            mentions = batch_df["text"].tolist()
            
            # Formulate GENRE input
            input_texts = [f"[START_ENT] {m} [END_ENT]" for m in mentions]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=32)
            
            decoded_entities = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Step 2: Wikidata Lookups (Slow over Network)
            for (_, row), entity in zip(batch_df.iterrows(), decoded_entities):
                qid = get_qid(entity)
                types = get_types(qid)
                
                writer.write({
                    "ner_id": row["id"], 
                    "el_id": qid,
                    "text": row["text"], 
                    "ner_label": row["label"],
                    "el_label": entity, 
                    "el_coarse_type": map_coarse(types),
                    "el_fine_types": types, 
                    "ner_score": float(row["score"])
                })
                done_keys.add(row["_el_key"])
            
            # Save progress every batch
            with open(EL_PROGRESS_PATH, "w") as f:
                json.dump(list(done_keys), f)

    # Export to CSV (Post-processing)
    if os.path.exists(EL_OUT_PATH):
        with jsonlines.open(EL_OUT_PATH) as reader:
            all_results = list(reader)
        if all_results:
            df_csv = pd.DataFrame(all_results)
            df_csv["el_fine_types"] = df_csv["el_fine_types"].apply(
                lambda x: x[0] if (isinstance(x, list) and len(x) > 0) else None
            )
            df_csv.to_csv(EL_CSV_PATH, index=False)

if __name__ == "__main__":
    main()