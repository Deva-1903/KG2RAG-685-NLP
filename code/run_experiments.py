#!/usr/bin/env python3
"""
Flexible Experiment Runner for KG²RAG

Runs both original and experimental pipelines with the same configuration.
Supports:
- Random or sequential sampling
- Any number of questions (100, 200, 300, etc.)
- Automatic seed management for random sampling
- Unique output filenames based on configuration
"""

import os
import sys
import subprocess
import argparse
import json
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


def generate_experiment_folder_name(num_questions, is_random, seed=None):
    """Generate experiment folder name based on configuration."""
    if is_random:
        folder_name = f"exp_{num_questions}_random_seed{seed}"
    else:
        folder_name = f"exp_{num_questions}_first"
    
    return folder_name


def run_experiment(num_questions, is_random=True, seed=None, base_output_dir="../output/hotpot"):
    """
    Run both original and experimental pipelines with same configuration.
    
    Args:
        num_questions: Number of questions to test
        is_random: If True, use random sampling; if False, use first N
        seed: Random seed (auto-generated if None and is_random=True)
        base_output_dir: Base directory for output files
    """
    # Auto-generate seed if not provided and using random sampling
    if is_random and seed is None:
        seed = 42  # Default seed
    
    # Create experiment folder
    experiment_folder_name = generate_experiment_folder_name(num_questions, is_random, seed)
    experiment_dir = os.path.join(base_output_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Generate output filenames (simple names inside folder)
    original_file = os.path.join(experiment_dir, "original.json")
    experimental_file = os.path.join(experiment_dir, "experimental.json")
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Number of questions: {num_questions}")
    print(f"Sampling method: {'Random' if is_random else 'First N'}")
    if is_random:
        print(f"Random seed: {seed}")
    print(f"Experiment folder: {experiment_dir}")
    print(f"Original output: {original_file}")
    print(f"Experimental output: {experimental_file}")
    print(f"{'='*80}\n")
    
    # Common arguments
    common_args = {
        'dataset': 'hotpotqa',
        'data_path': '../data/hotpotqa/hotpot_dev_distractor_v1.json',
        'kg_dir': '../data/hotpotqa/kgs/extract_subkgs',
        'num_questions': num_questions,
        'random_sample': is_random,
        'seed': seed if is_random else None,
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
    print("\n[1/2] Running Original KG²RAG Pipeline...")
    original_success = run_pipeline('kg_rag_distractor_100.py', original_args, original_file)
    
    if not original_success:
        print("\n❌ Original pipeline failed!")
        return False
    
    # Run experimental pipeline
    print("\n[2/2] Running Experimental KG²RAG Pipeline...")
    experimental_success = run_pipeline('kg_rag_enhanced_100.py', experimental_args, experimental_file)
    
    if not experimental_success:
        print("\n❌ Experimental pipeline failed!")
        return False
    
    # Save experiment metadata in the experiment folder
    metadata = {
        'num_questions': num_questions,
        'is_random': is_random,
        'seed': seed if is_random else None,
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
    print(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nResults saved in folder: {experiment_dir}")
    print(f"  Original: {os.path.basename(original_file)}")
    print(f"  Experimental: {os.path.basename(experimental_file)}")
    print(f"  Metadata: metadata.json")
    print(f"\nTo compare results, run:")
    print(f"  python generate_report.py --output_dir {experiment_dir}")
    print(f"\nOr generate report for all experiments:")
    print(f"  python generate_report.py --output_dir {base_output_dir}")
    print(f"{'='*80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run KG²RAG experiments with flexible configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run random 100 questions
  python run_experiments.py --num_questions 100 --random
  
  # Run first 200 questions
  python run_experiments.py --num_questions 200 --first
  
  # Run random 500 questions with specific seed
  python run_experiments.py --num_questions 500 --random --seed 123
  
  # Run multiple experiments
  python run_experiments.py --num_questions 100 200 300 --random
        """
    )
    
    parser.add_argument(
        '--num_questions',
        type=int,
        nargs='+',
        required=True,
        help='Number of questions to test (can specify multiple: 100 200 300)'
    )
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random sampling (default: False, uses first N)'
    )
    
    parser.add_argument(
        '--first',
        action='store_true',
        help='Use first N questions (default if --random not specified)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (auto: 42 if not specified and using random)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output/hotpot',
        help='Output directory for results (default: ../output/hotpot)'
    )
    
    args = parser.parse_args()
    
    # Determine sampling method
    # If --random is specified, use random
    # If --first is specified, use first N
    # If neither specified but seed is given, use random
    # Default: use first N
    if args.random:
        is_random = True
    elif args.first:
        is_random = False
    elif args.seed is not None:
        is_random = True  # Seed implies random sampling
    else:
        is_random = False  # Default to first N
    
    # Run experiments for each number of questions
    all_success = True
    for num_q in args.num_questions:
        print(f"\n{'#'*80}")
        print(f"# Running experiment: {num_q} questions ({'random' if is_random else 'first'})")
        print(f"{'#'*80}\n")
        
        success = run_experiment(
            num_questions=num_q,
            is_random=is_random,
            seed=args.seed,
            base_output_dir=args.output_dir
        )
        
        if not success:
            all_success = False
            print(f"\n⚠️  Experiment for {num_q} questions failed!")
        else:
            print(f"\n✅ Experiment for {num_q} questions completed!")
    
    if all_success:
        print(f"\n{'='*80}")
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠️  SOME EXPERIMENTS FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
