# A Fine-Grained Study of Long-Tail Entities in News Articles

## Project Overview

TODO

---

## Dataset

- **Signal-1M (Signal Media One Million News Articles)**  
  A large-scale news corpus commonly used for entity-centric research.  
  https://research.signal-ai.com/newsir16/signal-dataset.html

---

## 🧠 Methods

### Named Entity Recognition (NER)
Ideally, NER that can extract entities and has about 20-30 subtypes
# Default Subtypes / Taxonomy
# 🏷️ NER Taxonomy Reference Guide

> **Project:** Fine-Grained Entity Analysis
> **Purpose:** Reference documentation for Named Entity Recognition (NER) tag sets.

This document outlines the default "Subtypes" (Taxonomies) used by the major NER tools evaluated for this project. It contrasts **Coarse-Grained** industry standards (like spaCy) with **Fine-Grained** academic standards (like FIGER) and **Open-Vocabulary** modern tools (like GLiNER).

---

## ⚡ Quick Comparison

| Tool | Default Schema | # of Types | Granularity | Key Feature |
| :--- | :--- | :--- | :--- | :--- |
| **Stanford NER** | Stanford 3/4/7-Class | 3, 4, or 7 | Coarse | The "Classic" Java baseline. |
| **spaCy** | OntoNotes 5.0 | 18 | Coarse+ | Fast, Python-native industry standard. |
| **Stanza** | OntoNotes 5.0 | 18 | Coarse+ | High-accuracy standard NER. |
| **NameTag 3** | OntoNotes 5.0 (Eng) | 18 | Coarse+ | **SOTA** for *Nested* entities & Multilingual. |
| **FIGER** | FIGER System | 112 | Fine | Hierarchical (e.g., `/person/politician`). |
| **GLiNER** | **Open Vocabulary** | $\infty$ | **Custom** | Zero-shot; you define the list yourself. |

---

## 📚 Detailed Taxonomy Breakdown

### 1. Stanford NER (The Classic)
Stanford NER offers pre-trained models with 3 specific levels of granularity. It does not support fine-grained typing by default.

* **3-Class Model:** `PERSON`, `ORGANIZATION`, `LOCATION`
* **4-Class Model:** `PERSON`, `ORGANIZATION`, `LOCATION`, `MISC`
* **7-Class Model:**
    * `PERSON`
    * `ORGANIZATION`
    * `LOCATION`
    * `MONEY`
    * `PERCENT`
    * `DATE`
    * `TIME`

### 2. spaCy & Stanza (The Industry Standard)
Both tools utilize the **OntoNotes 5.0** corpus for their large English models (`en_core_web_trf` / `en`). This provides 18 flat categories.

**The 18 Types:**
* **PEOPLE:** `PERSON` (People, including fictional)
* **ORG:** `ORG` (Companies, agencies, institutions)
* **GROUPS:** `NORP` (Nationalities, religious or political groups)
* **GEOGRAPHY:** `GPE` (Countries, cities, states), `LOC` (Non-GPE: mountains, bodies of water)
* **STRUCTURES:** `FAC` (Buildings, airports, highways, bridges)
* **OBJECTS:** `PRODUCT` (Vehicles, foods, weapons - *not services*)
* **HAPPENINGS:** `EVENT` (Hurricanes, battles, wars, sports events)
* **ARTS:** `WORK_OF_ART` (Books, songs, paintings)
* **LEGAL:** `LAW` (Named documents made into laws)
* **LANG:** `LANGUAGE` (Any named language)
* **NUMERIC/TEMPORAL:** `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL`

### 3. NameTag 3 (The Modern SOTA)
[NameTag 3](https://github.com/ufal/nametag3) is a state-of-the-art tool (2024/2025) capable of **Nested NER** (detecting entities inside other entities).

**English Models:**
* **OntoNotes Model (Recommended):** Supports the same **18 types** listed above for spaCy.
* **CoNLL Model:** Supports the basic **4 types** (`PER`, `ORG`, `LOC`, `MISC`).

**Why use it?** Unlike spaCy, NameTag 3 can handle nested structures like:
> *"[University of [California]]"*
> * `University of California` -> `ORG`
> * `California` -> `GPE`

### 4. FIGER (The "A-Grade" Fine-Grained Standard)
FIGER is a dataset and taxonomy designed specifically for fine-grained analysis. It uses a **2-level hierarchy** with **112 types**.

**Common Examples (Level 1 / Level 2):**
* `/person` -> `/person/politician`, `/person/athlete`, `/person/actor`, `/person/author`
* `/organization` -> `/organization/company`, `/organization/educational_institution`, `/organization/sports_team`
* `/location` -> `/location/city`, `/location/province`, `/location/bridge`
* `/art` -> `/art/film`, `/art/music`
* `/product` -> `/product/software`, `/product/engine`
* `/building` -> `/building/hospital`, `/building/library`

### 5. GLiNER (The Zero-Shot Game Changer)
GLiNER does **not** have a default list. It uses a "Prompt-based" approach where you provide the list of labels at runtime.

**How to use it:**
You define a Python list with *any* strings you want:
```python
labels = ["Politician", "Start-up", "Algorithm", "Food Ingredient"]


### Entity Linking (EL)
- **BLINK (by Meta AI)**  
  Used to link detected entities to Wikipedia, based on fine-tuned BERT architectures
  https://github.com/facebookresearch/BLINK

---

## Output

- Detection of long-tail entities in news articles  
- Construction of a **fine-grained entity taxonomy**

---

## 📚 Related Work

- Esquivel, J., Albakour, D., Martinez, M., Corney, D., & Moussa, S.  
  *On the Long-Tail Entities in News*

---

## Tools

- Python  
- spaCy, Stanza, NameTag 3  
- BLINK (Meta AI)