# A Fine-Grained Study of Long-Tail Entities in News Articles

**Punreach Rany and Denilson Barbosa**  
Department of Computing Science, University of Alberta  
`{rany, denilson}@ualberta.ca`

---

## Overview

This repository contains the full pipeline for our study of long-tail entities in one million English news articles from the [Signal-1M](https://research.signal-ai.com/newsir16/signal-dataset.html) corpus. We use a 40-type NER schema with [GLiNER](https://github.com/urchade/GLiNER) and [ReFinED](https://github.com/amazon-science/ReFinED) to link entity mentions to Wikipedia, measuring how much of the entity space falls outside existing knowledge bases.

**Key finding:** 57–76% of unique named entity mentions in news text cannot be linked to Wikipedia, with the gap concentrated in Product, Event, and Miscellaneous types.

---

## Pipeline

```
Signal-1M (1M articles)
        │
        ▼
  [1] GLiNER NER — 40-type schema
      38.11M raw mentions
        │
        ▼
  [2] Combine + Deduplication
      5.4M unique (text, type) pairs
        │
        ▼
  [3] ReFinED Entity Linking → Wikipedia
        │
        ▼
  [4] Analysis
      Match / Mismatch / NIL / No Overlap
```

### Models

| Stage | Model |
|---|---|
| NER | `urchade/gliner_large-v2.1` |
| EL  | `ReFinED` (`wikipedia_model_with_numbers`) |

### Entity Type Schema

| Coarse Type | Fine-Grained Types |
|---|---|
| PERSON | Person, Politician, Athlete, Military Person, Religious Leader |
| LOCATION | GPE, Location, Facility |
| ORGANIZATION | Organization, Political Party, Government Agency, Sports Team, Website, Norp |
| EVENT | Event, Election, Military Conflict, Natural Disaster, Sports Event |
| PRODUCT | Product, Weapon, Vehicle, Disease, Chemical Thing, Living Thing |
| WORK OF ART | Work of Art, Film, Music, Written Work |
| NUMERIC | Date, Time, Percent, Money, Quantity, Ordinal, Cardinal |
| MISCELLANEOUS | Religion, Ethnicity, Language, Law |

---

## Repository Structure

```
.
├── split_data.py              # Split Signal-1M or NER output into shards
├── run_gliner.py              # GLiNER NER extraction (per shard)
├── combine.py                 # Combine per-shard NER or EL outputs
├── remove_duplicates.py       # Deduplicate (text, type) pairs
├── run_refined.py             # ReFinED EL (per shard)
├── constants.py               # Shared constants (LABELS list, etc.)
│
├── gliner_outputs/
│   ├── combined_gliner_output.csv   # All raw NER mentions (38.11M)
│   └── unique_gliner_output.csv     # Deduplicated NER mentions (5.4M)
│
├── unique_ner/                # Deduplicated NER shards (input to EL)
├── el_outputs/
│   └── combined_el_output.csv       # Combined EL output
│
├── analysis/
│   ├── ner_report.py              # Label count report from NER output
│   ├── count_type_frequency.py    # NER × EL type frequency table
│   └── error_analysis_sample.py   # Stratified 450-entity annotation sample
│
├── sample-1M.jsonl                # Signal-1M news dataset
├── paper.pdf                      # A Fine-Grained Study of Long-Tail Entities in News Articles.pdf
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```

---

## Reproducing Results

### Step 1 — Download the data

Download `sample-1M.jsonl` from [Signal-1M](https://research.signal-ai.com/newsir16/signal-dataset.html).

### Step 2 — Split into shards

```bash
python3 split_data.py -i sample-1M.jsonl -o splits
```

### Step 3 — Run GLiNER NER (repeat for shards 1–10)

```bash
# Optional: set your Hugging Face token to speed up model download
export HF_TOKEN=your_token_here

python3 run_gliner.py -s 1 -i splits -o gliner_outputs
# repeat with -s 2, 3, ... 10
```

### Step 4 — Combine and deduplicate NER

```bash
python3 combine.py -f gliner_outputs -o gliner_outputs/combined_gliner_output.csv

python3 remove_duplicates.py \
  -i gliner_outputs/combined_gliner_output.csv \
  -o gliner_outputs/unique_gliner_output.csv
```

### Step 5 — Split deduplicated NER output for EL

```bash
python3 split_data.py -i gliner_outputs/unique_gliner_output.csv -o unique_ner
```

### Step 6 — Run ReFinED EL (repeat for shards 1–10)

```bash
python3 run_refined.py -t 1 -i unique_ner -o el_outputs
# repeat with -t 2, 3, ... 10
```

### Step 7 — Combine EL outputs

```bash
python3 combine.py -f el_outputs -o el_outputs/combined_el_output.csv
```

---

## Analysis

All analysis scripts live in the `analysis/` folder.

**NER label report** — counts how many times each entity type was extracted:
```bash
python3 analysis/ner_report.py \
  -i gliner_outputs/combined_gliner_output.csv \
  -o gliner_outputs/combined_ner_report.csv
```

**NER × EL frequency table** — for each NER type, counts how many entities were linked to each EL type (PERSON, ORG, GPE, etc.), NIL, or Unlinkable:
```bash
python3 analysis/count_type_frequency.py \
  -i el_outputs/combined_el_output.csv \
  -o ner_to_el_frequencies.csv
```

**Error analysis sample** — stratified sample of 50 entities per NER × EL type cell (3×3 grid = 450 total) for manual annotation:
```bash
python3 analysis/error_analysis_sample.py \
  -i el_outputs/combined_el_output.csv \
  -o error_analysis_sample.csv \
  -n 50
```

The output CSV includes `correct` and `notes` columns for manual annotation. Each row should be labelled as **Both Correct**, **NER Correct** (NER label right, EL entity wrong), or **EL Correct** (EL entity right, NER label wrong).

---

## Key Results

| Coarse Type | Total | Match (%) | NIL (%) | No Overlap (%) |
|---|---|---|---|---|
| Location | 654,449 | 23.83 | 12.21 | 57.32 |
| Person | 1,478,994 | 22.49 | 4.55 | 68.01 |
| Work of Art | 247,613 | 18.58 | 16.00 | 57.82 |
| Organization | 1,313,672 | 13.40 | 12.99 | 67.42 |
| Event | 316,419 | 4.34 | 15.44 | 75.85 |
| Miscellaneous | 157,669 | 0.57 | 24.09 | 69.81 |
| Product | 601,014 | 0.01 | 21.58 | 71.70 |

Both NIL and No Overlap are treated as long-tail signals. NIL entities exist in Wikipedia but lack a Wikidata type; No Overlap entities are absent from Wikipedia entirely.

---

## License

This project is released for academic use.  
The Signal-1M dataset is subject to its own [terms of use](https://research.signal-ai.com/datasets/signal1m-tweetir.html).