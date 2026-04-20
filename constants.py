# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "sample-1M.jsonl"
SPLIT_DATASET_PATH = "splits"  # optional: for chunking long articles
NER_OUT_PATH   = "ner_results.jsonl"        # one JSON line per article
PROGRESS_PATH  = "ner_progress.json"        # set of completed article IDs
CSV_OUT_PATH = "ner_results.csv"

# ── GLiNER labels ────────────────────────────────────────────────────────────
LABELS = [
    # ── PERSON → ReFinED: PERSON ──────────────────────────────────────────────
    "person",            # OntoNotes 5.0
    "politician",        # FIGER
    "athlete",           # FIGER
    "military_person",   # FIGER
    "religious_leader",  # FIGER

    # ── ORGANIZATION → ReFinED: ORG ───────────────────────────────────────────
    "organization",      # OntoNotes 5.0
    "political_party",   # FIGER
    "government_agency", # FIGER
    "sports_team",       # FIGER
    "website",           # FIGER → ReFinED: ORG

    # ── NORP → ReFinED: no direct output (expect NIL / PERSON / ORG) ──────────
    "norp",              # OntoNotes 5.0
    "religion",          # FIGER
    "ethnicity",         # FIGER

    # ── LOCATION → ReFinED: GPE or FAC ────────────────────────────────────────
    "gpe",               # OntoNotes 5.0 → ReFinED: GPE
    "location",          # OntoNotes 5.0 → ReFinED: GPE
    "facility",          # OntoNotes 5.0 → ReFinED: FAC

    # ── EVENT → ReFinED: EVENT ────────────────────────────────────────────────
    "event",             # OntoNotes 5.0
    "election",          # FIGER
    "military_conflict", # FIGER
    "natural_disaster",  # FIGER
    "sports_event",      # FIGER

    # ── WORK_OF_ART → ReFinED: WORK_OF_ART ───────────────────────────────────
    "work_of_art",       # OntoNotes 5.0
    "film",              # FIGER
    "music",             # FIGER
    "written_work",      # FIGER

    # ── PRODUCT → ReFinED: PRODUCT ────────────────────────────────────────────
    "product",           # OntoNotes 5.0
    "weapon",            # FIGER
    "vehicle",           # FIGER
    "disease",           # FIGER — expected high NIL
    "chemical_thing",    # FIGER — expected high NIL
    "living_thing",      # FIGER — expected near-100% no-overlap

    # ── LANGUAGE → ReFinED: LANGUAGE ──────────────────────────────────────────
    "language",          # OntoNotes 5.0

    # ── LAW → ReFinED: ORG ────────────────────────────────────────────────────
    "law",               # OntoNotes 5.0

    # ── Numeric → ReFinED outputs these directly ──────────────────────────────
    "date",              # OntoNotes 5.0 → ReFinED: DATE
    "time",              # OntoNotes 5.0 → ReFinED: TIME
    "percent",           # OntoNotes 5.0 → ReFinED: PERCENT
    "money",             # OntoNotes 5.0 → ReFinED: MONEY
    "quantity",          # OntoNotes 5.0 → ReFinED: QUANTITY
    "ordinal",           # OntoNotes 5.0 → ReFinED: ORDINAL
    "cardinal",          # OntoNotes 5.0 → ReFinED: CARDINAL
]

# ── Chunking ─────────────────────────────────────────────────────────────────
# GLiNER ~380 subword tokens ≈ ~280 words safely
# ── Section 0 config update ───────────────────────────────────────────────────
GLINER_MAX_TOKENS    = 380   # lowered from 380 to absorb GLiNER's internal specials
GLINER_STRIDE_TOKENS = 60
WARNINGS_PATH        = "ner_warnings.json"   # NEW: articles that triggered truncation

GLINER_THRESHOLD  = 0.4   # entity confidence threshold
