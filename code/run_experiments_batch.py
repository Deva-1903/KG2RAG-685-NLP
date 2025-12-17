#!/usr/bin/env python3
"""
Batch Experiment Runner - Ensures Non-Overlapping Question Sets

For distributed experiments across multiple teammates, this script ensures
each teammate gets a unique, non-overlapping set of questions.

Usage:
  # Teammate 1: Gets questions 0-2500
  python run_experiments_batch.py --num_questions 2500 --batch_id 0 --total_batches 3
  
  # Teammate 2: Gets questions 2500-5000
  python run_experiments_batch.py --num_questions 2500 --batch_id 1 --total_batches 3
  
  # Teammate 3: Gets questions 5000-7500
  python run_experiments_batch.py --num_questions 2500 --batch_id 2 --total_batches 3
"""

import os
import sys
import subprocess
import argparse
import json
import random
from pathlib import Path


def run_pipeline(script_name, args_dict, output_file):
    """Run a pipeline script with given arguments."""
    cmd = ['python', script_name]
    
    for key, value in args_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.append(f'--{key}')
                cmd.append(str(value))
    
    # Override result path
    cmd.append('--result_path')
    cmd.append(output_file)
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def generate_experiment_folder_name(num_questions, batch_id, total_batches):
    """Generate experiment folder name based on batch configuration."""
    folder_name = f"exp_{num_questions}_batch{batch_id}of{total_batches}"
    return folder_name


def run_batch_experiment(num_questions, batch_id, total_batches, base_output_dir="../output/hotpot", seed=None):
    """
    Run both pipelines on a specific batch of questions (non-overlapping).
    
    Args:
        num_questions: Number of questions per batch
        batch_id: Batch ID (0-indexed: 0, 1, 2, ...)
        total_batches: Total number of batches
        base_output_dir: Base directory for output files
        seed: Optional seed for shuffling (same seed = same shuffle order)
    """
    # Create experiment folder
    experiment_folder_name = generate_experiment_folder_name(num_questions, batch_id, total_batches)
    experiment_dir = os.path.join(base_output_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Generate output filenames
    original_file = os.path.join(experiment_dir, "original.json")
    experimental_file = os.path.join(experiment_dir, "experimental.json")
    
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Questions per batch: {num_questions}")
    print(f"Batch ID: {batch_id} (of {total_batches} total batches)")
    print(f"Question range: {batch_id * num_questions} to {(batch_id + 1) * num_questions}")
    if seed:
        print(f"Shuffle seed: {seed} (for consistent ordering)")
    print(f"Experiment folder: {experiment_dir}")
    print(f"Original output: {original_file}")
    print(f"Experimental output: {experimental_file}")
    print(f"{'='*80}\n")
    
    # Calculate question range
    start_idx = batch_id * num_questions
    end_idx = (batch_id + 1) * num_questions
    
    # Common arguments
    common_args = {
        'dataset': 'hotpotqa',
        'data_path': '../data/hotpotqa/hotpot_dev_distractor_v1.json',
        'kg_dir': '../data/hotpotqa/kgs/extract_subkgs',
        'num_questions': num_questions,
        'random_sample': False,  # We'll use sequential range
        'seed': None,
        'start_idx': start_idx,  # Custom parameter for batch range
    }
    
    # Original pipeline arguments
    original_args = {
        **common_args,
        'model_name': 'llama3:8b',
        'embed_model_name': 'mxbai-embed-large',
        'reranker': '../model/bge-reranker-large',
        'top_k': 10,
        'use_tpt': False,
    }
    
    # Experimental pipeline arguments
    experimental_args = {
        **common_args,
        'model_name': 'llama3:8b',
        'embed_model_name': 'mxbai-embed-large',
        'top_k': 10,
        'fusion_method': 'rrf',
        'top_n_per_view': 10,
        'seed_top_k': 8,
        'token_budget': 2048,
        'use_dp_knapsack': False,
    }
    
    # Run original pipeline
    print(f"\n[1/2] Running Original KG²RAG Pipeline (Batch {batch_id})...")
    original_success = run_pipeline('kg_rag_distractor_100.py', original_args, original_file)
    
    if not original_success:
        print("\n❌ Original pipeline failed!")
        return False
    
    # Run experimental pipeline
    print(f"\n[2/2] Running Experimental KG²RAG Pipeline (Batch {batch_id})...")
    experimental_success = run_pipeline('kg_rag_enhanced_100.py', experimental_args, experimental_file)
    
    if not experimental_success:
        print("\n❌ Experimental pipeline failed!")
        return False
    
    # Save experiment metadata
    metadata = {
        'num_questions': num_questions,
        'batch_id': batch_id,
        'total_batches': total_batches,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'experiment_folder': experiment_dir,
        'original_output': original_file,
        'experimental_output': experimental_file,
        'original_detailed': original_file.replace('.json', '_detailed.json'),
        'experimental_detailed': experimental_file.replace('.json', '_detailed.json'),
    }
    
    metadata_file = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH {batch_id} COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nResults saved in folder: {experiment_dir}")
    print(f"  Questions tested: {start_idx} to {end_idx}")
    print(f"  Original: {os.path.basename(original_file)}")
    print(f"  Experimental: {os.path.basename(experimental_file)}")
    print(f"\nAfter all batches complete, compile with:")
    print(f"  python compile_batches.py --batch_dirs {experiment_folder_name} ...")
    print(f"{'='*80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run batch experiments with non-overlapping question sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Teammate 1: First batch (questions 0-2500)
  python run_experiments_batch.py --num_questions 2500 --batch_id 0 --total_batches 3
  
  # Teammate 2: Second batch (questions 2500-5000)
  python run_experiments_batch.py --num_questions 2500 --batch_id 1 --total_batches 3
  
  # Teammate 3: Third batch (questions 5000-7500)
  python run_experiments_batch.py --num_questions 2500 --batch_id 2 --total_batches 3
        """
    )
    
    parser.add_argument(
        '--num_questions',
        type=int,
        required=True,
        help='Number of questions per batch'
    )
    
    parser.add_argument(
        '--batch_id',
        type=int,
        required=True,
        help='Batch ID (0-indexed: 0, 1, 2, ...)'
    )
    
    parser.add_argument(
        '--total_batches',
        type=int,
        required=True,
        help='Total number of batches'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional seed for shuffling dataset before splitting (ensures consistent ordering)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output/hotpot',
        help='Output directory for results (default: ../output/hotpot)'
    )
    
    args = parser.parse_args()
    
    # Validate batch_id
    if args.batch_id < 0 or args.batch_id >= args.total_batches:
        print(f"❌ Error: batch_id must be between 0 and {args.total_batches - 1}")
        sys.exit(1)
    
    # Run batch experiment
    success = run_batch_experiment(
        num_questions=args.num_questions,
        batch_id=args.batch_id,
        total_batches=args.total_batches,
        base_output_dir=args.output_dir,
        seed=args.seed
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
