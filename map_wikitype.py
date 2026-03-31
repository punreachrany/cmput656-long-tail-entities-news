import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os

# 1. Your predefined labels
LABELS = [
    "person", "norp", "facility", "organization", "gpe", "location",
    "product", "event", "work_of_art", "law", "language", "date", "time", 
    "percent", "money", "quantity", "ordinal", "cardinal", "religion", 
    "political_party", "nationality", "ethnic_group", "title", "award", 
    "disease", "chemical", "weapon", "vehicle", "currency", "brand"
]

# 2. Load a lightweight, super-fast bi-encoder model
# This uses the exact same matching philosophy as GLiNER
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Sentence Transformer on {device}...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 3. Pre-compute the embeddings for your 30 labels (happens instantly)
print("Embedding labels...")
label_embeddings = model.encode(LABELS, convert_to_tensor=True)

# 4. Load your CSV
CSV_PATH = "tests/master_fine_grain_types.csv"
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: File '{CSV_PATH}' not found.")
    exit(1)

df = pd.read_csv(CSV_PATH)
unique_types = df['el_fine_type'].dropna().unique().tolist()
print(f"Found {len(unique_types):,} unique fine-grained types.")

mapping_dict = {}

# 5. Classify using Cosine Similarity
print("Mapping unique types to predefined labels...")
# We can process them in one giant batch because Sentence Transformers are so fast
type_embeddings = model.encode(unique_types, convert_to_tensor=True, show_progress_bar=True)

# Calculate similarity between all types and all labels
cosine_scores = util.cos_sim(type_embeddings, label_embeddings)

# Find the highest scoring label for each type
for i in range(len(unique_types)):
    best_label_idx = torch.argmax(cosine_scores[i]).item()
    mapping_dict[unique_types[i]] = LABELS[best_label_idx]

# 6. Save the mapping
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=['el_fine_type', 'mapped_label'])
mapping_df.to_csv("tests/type_mapping_dictionary_st.csv", index=False)
print("\n✅ Mapping saved to type_mapping_dictionary_st.csv")