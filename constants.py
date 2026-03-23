# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "sample-1M.jsonl"
SPLIT_DATASET_PATH = "splits"  # optional: for chunking long articles
NER_OUT_PATH   = "ner_results.jsonl"        # one JSON line per article
PROGRESS_PATH  = "ner_progress.json"        # set of completed article IDs
CSV_OUT_PATH = "ner_results.csv"

# ── GLiNER labels ────────────────────────────────────────────────────────────
LABELS = [
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language",
    "date", "time", "percent", "money", "quantity", "ordinal", "cardinal",

    
    "religion", "political_party", "nationality", "ethnic_group",
    "title", "award", "disease", "chemical", "weapon",
    "vehicle", "currency", "brand"
]

# ── Chunking ─────────────────────────────────────────────────────────────────
# GLiNER ~380 subword tokens ≈ ~280 words safely
# ── Section 0 config update ───────────────────────────────────────────────────
GLINER_MAX_TOKENS    = 380   # lowered from 380 to absorb GLiNER's internal specials
GLINER_STRIDE_TOKENS = 60
WARNINGS_PATH        = "ner_warnings.json"   # NEW: articles that triggered truncation

GLINER_THRESHOLD  = 0.4   # entity confidence threshold
