# A Fine-Grained Study of Long-Tail Entities in News Articles

**Punreach Rany and Denilson Barbosa**  
Department of Computing Science, University of Alberta  
`{rany, denilson}@ualberta.ca`

---

## Overview

This repository contains the code and data for our study of long-tail entities in one million English news articles from the [Signal-1M](https://research.signal-ai.com/datasets/signal1m-tweetir.html) corpus.

We find that **57–76% of unique named entity mentions in news text cannot be linked to Wikipedia**, and that this gap is structured by entity type — concentrated in Product, Event, and Miscellaneous types, with fine-grained rates ranging from 52% (Politician) to 97% (Law).

---

## Pipeline

```
Signal-1M corpus (1M articles)
        │
        ▼
  GLiNER NER (40-type schema)
  38.11M raw mentions
        │
        ▼
  Deduplication on (text, type) pairs
  5.4M unique entities
        │
        ▼
  ReFinED Entity Linking → Wikipedia
        │
        ▼
  Outcome classification: Match / Mismatch / NIL / No Overlap
```

### Models

| Stage | Model | Notes |
|---|---|---|
| NER | `urchade/gliner_large-v2.1` | Zero-shot, 40-type custom schema |
| EL | `ReFinED` (`wikipedia_model_with_numbers`) | Joint typing + disambiguation |

### Entity Type Schema

We use a unified 40-type schema combining OntoNotes 5.0 (18 types) and a 22-type subset of FIGER:

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
├── ner/
│   ├── run_ner.py              # GLiNER extraction pipeline
│   └── ner_outputs/            # Per-shard NER outputs (CSV)
│
├── el/
│   ├── perform_el.py           # ReFinED EL pipeline (SLURM array job)
│   ├── perform_el.sh           # SLURM job script
│   └── el_raw_outputs/         # Per-shard EL outputs (JSONL + CSV)
│
├── analysis/
│   ├── combine_dedup.py        # Combine + deduplicate NER outputs
│   ├── coarse_stats.py         # Compute coarse/fine-grained outcome tables
│   ├── dedup_comparison.py     # Reproduce prior work NER + dedup comparison
│   └── error_analysis_sample.py # Stratified sampling for manual annotation
│
├── data/
│   ├── unique_el_final/
│   │   └── combined_el_output_refined.csv   # Final 5.4M unique linked entities
│   └── error_analysis_sample.csv            # 450-entity annotation sample
│
└── README.md
```

---

## Setup

### Requirements

```bash
pip install gliner pandas torch transformers
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```

---


## Reproducing Results

Step 1: Download sample-1M.jsonl from signal-1M
Step 2: Run pip install requirement.txt
Step 2: Run python3 split_data.py -i sample-1M.jsonl -o splits
Step 3: Import Hugging Face Token (Optional : But this will speedup Gliner)
Step 3: Run run_gliner.py 0 to 10
Step 4: Run python combine.py -f gliner_outputs -o combined_gliner_output.csv
Step 5: Run python remove_duplicates.py -i gliner_outputs/combined_gliner_output.csv -o gliner_outputs/unique_gliner_output.csv
Step 6: Run run_refined.py 0 to 10

### 1. NER Extraction

```bash
python ner/run_ner.py
# Outputs: ner_outputs/ner_output_{0-9}.csv
# Progress saved to: ner_progress.json
```

### 2. Deduplication

```bash
python analysis/combine_dedup.py
# Output: combine_unique_ner.csv (5.4M unique pairs)
```

### 3. Entity Linking

```bash
# Local (parallel, 4 workers)
python el/run_parallel.py

# Compute Canada (SLURM array, tasks 0-9)
sbatch el/perform_el.sh
```

### 4. Analysis

```bash
# Coarse and fine-grained outcome tables
python analysis/coarse_stats.py

# Deduplication comparison with Esquivel et al. (2017)
python analysis/dedup_comparison.py

# Error analysis sample (450 entities, 3×3 NER×EL grid)
python analysis/error_analysis_sample.py
```

---

## Key Results

| Coarse Type | Total | Match (%) | NIL (%) | No Overlap (%) |
|---|---|---|---|---|
| Location | 654,449 | 23.83 | 12.21 | 57.32 |
| Person | 1,478,994 | 22.49 | 4.55 | 68.01 |
| Work of Art | 247,613 | 18.58 | 16.00 | 57.82 |
| Organization | 1,313,672 | 13.40 | 12.99 | 67.42 |
| Event | 316,419 | 4.34 | 15.44 | 75.85 |
| Product | 601,014 | 0.01 | 21.58 | 71.70 |

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{rany2025longtail,
  title     = {A Fine-Grained Study of Long-Tail Entities in News Articles},
  author    = {Rany, Punreach and Barbosa, Denilson},
  year      = {2025},
  institution = {University of Alberta}
}
```

---

## License

This project is for academic use. The Signal-1M dataset is subject to its own [terms of use](https://research.signal-ai.com/datasets/signal1m-tweetir.html).