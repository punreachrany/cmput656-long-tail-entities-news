import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os

# 1. Your 30 predefined NER labels (The items we want to categorize)
NER_LABELS = [
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language", "date", "time", 
    "percent", "money", "quantity", "ordinal", "cardinal", "religion", 
    "political_party", "nationality", "ethnic_group", "title", "award", 
    "disease", "chemical", "weapon", "vehicle", "currency", "brand"
]

# 2. ReFinED's 15 coarse types (The buckets we are mapping INTO)
EL_TYPES = [
    "PERSON", "WORK_OF_ART", "GPE", "ORG", "FAC", "DATE", 
    "LANGUAGE", "CARDINAL", "PRODUCT", "EVENT", "PERCENT", 
    "TIME", "ORDINAL", "QUANTITY", "MONEY"
]

# Clean up EL_TYPES for better semantic embedding (e.g., "WORK_OF_ART" -> "work of art")
# We only use this list for the math, the final output will still use the uppercase ones.
cleaned_el_types = [t.lower().replace("_", " ") for t in EL_TYPES]

# 3. Load the Sentence Transformer
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Sentence Transformer on {device}...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 4. Embed both lists
print("Embedding ReFinED EL Types...")
el_type_embeddings = model.encode(cleaned_el_types, convert_to_tensor=True)

print("Embedding GLiNER NER Labels...")
ner_label_embeddings = model.encode(NER_LABELS, convert_to_tensor=True)

# 5. Classify using Cosine Similarity
print("Calculating similarities...")
cosine_scores = util.cos_sim(ner_label_embeddings, el_type_embeddings)

mapping_dict = {}

# Find the highest scoring EL_TYPE for each NER_LABEL
for i in range(len(NER_LABELS)):
    best_idx = torch.argmax(cosine_scores[i]).item()
    mapping_dict[NER_LABELS[i]] = EL_TYPES[best_idx]

# 6. Save the mapping
os.makedirs("tests", exist_ok=True)
output_path = "tests/ner_to_el_mapping.csv"

mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=['ner_label', 'mapped_el_type'])
mapping_df.to_csv(output_path, index=False)

print(f"\n✅ Mapping complete! Saved to {output_path}")

# Print a quick preview
print("\n--- Mapping Preview ---")
print(mapping_df.head(10).to_string(index=False))