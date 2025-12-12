#!/usr/bin/env python3
"""
KG Extraction script that processes a specific part of the dataset.

SIMPLE USAGE:
1. Set PART_NUMBER below (1, 2, 3, 4, or 5)
2. Run: python hotpot_extraction_part.py

Each teammate runs this with a different PART_NUMBER.
"""

import os
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

# ============================================================================
# CONFIGURATION - CHANGE THIS NUMBER FOR YOUR PART
# ============================================================================
PART_NUMBER = 1  # Change this to 1, 2, 3, 4, or 5 for your assigned part
# ============================================================================

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

# Configuration
DATA_DIR = '../../data/hotpotqa/splits'
OUT_DIR = '../../data/hotpotqa/kgs/extract_subkgs'

# Construct data path
data_path = os.path.join(DATA_DIR, f'hotpot_dev_distractor_v1_part{PART_NUMBER}.json')

# Validate part number
if PART_NUMBER < 1 or PART_NUMBER > 5:
    print(f"✗ Error: PART_NUMBER must be between 1 and 5. You set: {PART_NUMBER}")
    exit(1)

# Check if split file exists
if not os.path.exists(data_path):
    print(f"✗ Error: File not found: {data_path}")
    print(f"  Make sure you've run split_dataset.py first to create the split files!")
    print(f"  Expected file: {data_path}")
    exit(1)

print("=" * 80)
print(f"Processing Part {PART_NUMBER} of dataset")
print("=" * 80)
print(f"Data file: {data_path}")
print(f"Output directory: {OUT_DIR}")
print()

# Load data
with open(data_path) as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions from part {PART_NUMBER}")
print(f"Estimated time: ~{len(data) * 0.5 / 60:.1f} hours (with GPU) or ~{len(data) * 1.5 / 60:.1f} hours (CPU only)")
print()

# Setup
triplets = {}
llm = Ollama(model='llama3:8b', request_timeout=120)
os.makedirs(OUT_DIR, exist_ok=True)
count = 0

# Process
print("Starting extraction...")
for sample in tqdm(data, desc=f"Part {PART_NUMBER}"):
    question = sample['question']
    answer = sample['answer']
    ctxs = sample['context']
    for ctx in ctxs:
        ent = ctx[0]
        if ent in triplets:
            continue
        out_path = os.path.join(OUT_DIR, f'{ent.replace("/","_")}.json')
        if os.path.exists(out_path):
            continue
        triplets[ent] = {}
        for i in range(len(ctx[1])):
            if not i==0:
                ctx_text = f'{ent}: {ctx[1][i]}'
            else:
                ctx_text = ctx[1][i]
            ext_triplets = extract_triplets(llm, ctx_text)
            if len(ext_triplets)==0:
                continue
            triplets[ent][i] = ext_triplets
        with open(out_path,'w') as f:
            json.dump(triplets[ent], f)
        count += 1

print()
print("=" * 80)
print(f"✅ Part {PART_NUMBER} complete!")
print("=" * 80)
print(f'Newly extracted entity KGs: {count}')
print(f'All KGs saved to: {OUT_DIR}')
print()
print("Next steps:")
print("1. Share the KG files with your team (or upload to shared drive)")
print("2. Combine all parts when everyone is done")
print("3. Verify all files are in the extract_subkgs directory")
