#!/usr/bin/env python3
"""
Test script to process a single question from the dataset.
This is useful for testing and debugging the KG²RAG pipeline.
"""

import os
import sys
import ujson as json
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_rag_distractor import (
    read_data, init_model, read_kg, process_sample
)

def test_single_question(question_index=0, dataset='hotpotqa'):
    """
    Test the KG²RAG pipeline on a single question.
    
    Args:
        question_index: Index of the question to test (default: 0)
        dataset: Dataset name ('hotpotqa' or 'musique')
    """
    
    # Set up paths
    if dataset == 'hotpotqa':
        data_path = '../data/hotpotqa/hotpot_dev_distractor_v1.json'
        kg_dir = '../data/hotpotqa/kgs/extract_subkgs'
    elif dataset == 'musique':
        data_path = '../data/MuSiQue/musique_ans_v1.0_dev_mapped.jsonl'
        kg_dir = '../data/MuSiQue/kgs/extract_subkgs'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create args object
    args = argparse.Namespace()
    args.dataset = dataset
    args.data_path = data_path
    args.kg_dir = kg_dir
    args.model_name = 'llama3:8b'
    args.embed_model_name = 'mxbai-embed-large'
    args.reranker = '../model/bge-reranker-large'
    args.top_k = 10
    args.use_tpt = False
    
    print("=" * 80)
    print("KG²RAG Single Question Test")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/4] Loading dataset...")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Please ensure the dataset file exists.")
        return
    
    data = read_data(args)
    print(f"✓ Loaded {len(data)} questions from dataset")
    
    if question_index >= len(data):
        print(f"ERROR: Question index {question_index} is out of range (max: {len(data)-1})")
        return
    
    sample = data[question_index]
    
    # Display question info
    if dataset == 'hotpotqa':
        question = sample['question']
        answer = sample['answer']
        sample_id = sample['_id']
        print(f"\nQuestion ID: {sample_id}")
    else:
        question = sample['question']
        answer = sample['answer']
        sample_id = sample['id']
        print(f"\nQuestion ID: {sample_id}")
    
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {answer}")
    
    # Step 2: Initialize models
    print("\n[2/4] Initializing models...")
    try:
        init_model(args)
        print("✓ Models initialized (Ollama must be running)")
    except Exception as e:
        print(f"ERROR: Failed to initialize models: {e}")
        print("Make sure Ollama is running and models are downloaded:")
        print("  - ollama pull llama3:8b")
        print("  - ollama pull mxbai-embed-large")
        return
    
    # Step 3: Load knowledge graphs
    print("\n[3/4] Loading knowledge graphs...")
    if not os.path.exists(kg_dir):
        print(f"WARNING: KG directory not found: {kg_dir}")
        print("You may need to run preprocessing first:")
        print(f"  cd preprocess && python {dataset}_extraction.py")
        kg = {}
    else:
        kg = read_kg(args, data)
        print(f"✓ Loaded KGs for {len(kg)} entities")
    
    # Step 4: Process the question
    print("\n[4/4] Processing question...")
    print("-" * 80)
    
    try:
        sample_id, prediction, sps = process_sample(args, sample, kg)
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"\nPredicted Answer: {prediction}")
        print(f"Ground Truth: {answer}")
        print(f"\nSupporting Facts: {sps}")
        print(f"Number of supporting facts: {len(sps)}")
        
        # Check if answer matches
        if prediction.lower().strip() == answer.lower().strip():
            print("\n✓ Answer matches ground truth!")
        else:
            print("\n✗ Answer does not match ground truth")
        
    except Exception as e:
        print(f"\nERROR: Failed to process question: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test KG²RAG on a single question')
    parser.add_argument('--index', type=int, default=0, 
                       help='Index of question to test (default: 0)')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                       choices=['hotpotqa', 'musique'],
                       help='Dataset to use (default: hotpotqa)')
    
    args = parser.parse_args()
    
    test_single_question(question_index=args.index, dataset=args.dataset)

