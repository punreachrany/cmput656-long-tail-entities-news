import json
import os
import time

# Update these paths to match your filenames
INPUT_FILE = "new_articles_2025.deduped"
OUTPUT_FILE = "new_articles_2025.jsonl"

print(f"🚀 Starting full conversion of {INPUT_FILE}...")
start_time = time.time()

count = 0
# Use a buffer size for faster writing on cluster storage
with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8', buffering=1024*1024) as f_out:
    
    for line in f_in:
        clean_text = line.strip()
        if not clean_text:
            continue
            
        # Create the dictionary
        article = {
            "article_id": f"wmt23_{count}",
            "text": clean_text,
            "source": "WMT23_Crawl"
        }
        
        # Write to JSONL
        f_out.write(json.dumps(article) + "\n")
        
        count += 1
        
        # Progress log every 500k rows
        if count % 500_000 == 0:
            elapsed = time.time() - start_time
            print(f"📦 Processed {count:,} articles... ({count/elapsed:.1f} art/sec)")

total_time = time.time() - start_time
print(f"✅ COMPLETE!")
print(f"Final Count: {count:,} articles")
print(f"Output File: {OUTPUT_FILE}")
print(f"Total Time:  {total_time/60:.2f} minutes")