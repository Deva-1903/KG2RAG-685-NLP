import sys
sys.path.append('code')

from code.kg_rag_distractor import *
import argparse

# Create minimal args
args = argparse.Namespace()
args.dataset = 'hotpotqa'
args.data_path = '../data/hotpotqa/hotpot_dev_distractor_v1.json'
args.kg_dir = '../data/hotpotqa/kgs/extract_subkgs'
args.model_name = 'llama3:8b'
args.embed_model_name = 'mxbai-embed-large'
args.reranker = '../model/bge-reranker-large'
args.top_k = 10
args.use_tpt = False

# Load data
data = read_data(args)
kg = read_kg(args, data)

# Process just the first question
sample = data[0]
sample_id, prediction, sps = process_sample(args, sample, kg)

print(f"Question: {sample['question']}")
print(f"Answer: {prediction}")
print(f"Supporting facts: {sps}")