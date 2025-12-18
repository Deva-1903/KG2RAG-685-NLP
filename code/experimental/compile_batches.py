#!/usr/bin/env python3
"""
Batch Compilation Script for Distributed Experiments

Allows multiple teammates to run experiments in batches and compile results together.
Supports averaging metrics across batches and generating unified reports.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics
import math


def load_batch_result(batch_dir: str, pipeline_type: str = 'original') -> Dict:
    """
    Load results from a batch experiment folder.
    
    Args:
        batch_dir: Path to batch experiment folder (e.g., exp_2500_random_seed42)
        pipeline_type: 'original' or 'experimental'
    
    Returns:
        Dictionary with predictions, metadata, and tested questions
    """
    batch_path = Path(batch_dir)
    
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    
    # Load predictions
    pred_file = batch_path / f"{pipeline_type}.json"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Load detailed results (for metadata and tested questions)
    detailed_file = batch_path / f"{pipeline_type}_detailed.json"
    metadata = {}
    tested_questions = []
    
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            detailed = json.load(f)
            metadata = detailed.get('metadata', [])
            tested_questions = detailed.get('summary', {}).get('questions_tested', [])
    
    # Load metadata file if separate
    metadata_file = batch_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata_data = json.load(f)
            if isinstance(metadata_data, dict) and 'metadata' in metadata_data:
                metadata = metadata_data['metadata']
    
    return {
        'predictions': predictions,
        'metadata': metadata,
        'tested_questions': tested_questions,
        'batch_dir': str(batch_path)
    }


def merge_predictions(batch_results: List[Dict]) -> Dict:
    """Merge predictions from multiple batches."""
    merged_predictions = {'answer': {}, 'sp': {}}
    all_tested_questions = []
    all_metadata = []
    
    for batch in batch_results:
        preds = batch['predictions']
        
        # Merge answers
        if 'answer' in preds:
            merged_predictions['answer'].update(preds['answer'])
        
        # Merge supporting facts
        if 'sp' in preds:
            merged_predictions['sp'].update(preds['sp'])
        
        # Collect tested questions
        all_tested_questions.extend(batch['tested_questions'])
        
        # Collect metadata
        if isinstance(batch['metadata'], list):
            all_metadata.extend(batch['metadata'])
        elif isinstance(batch['metadata'], dict):
            all_metadata.append(batch['metadata'])
    
    return {
        'predictions': merged_predictions,
        'tested_questions': all_tested_questions,
        'metadata': all_metadata
    }


def calculate_batch_statistics(batch_results: List[Dict], gold_file: str) -> Dict:
    """
    Calculate statistics across batches (for metrics that can be averaged).
    
    This is useful for understanding variance across batches.
    """
    # Load gold data
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)
    gold_lookup = {dp['_id']: dp for dp in gold_data}
    
    # For each batch, calculate basic metrics
    batch_metrics = []
    
    for batch in batch_results:
        preds = batch['predictions']
        total = 0
        exact_matches = 0
        
        for qid, answer in preds.get('answer', {}).items():
            if qid in gold_lookup:
                total += 1
                gold_answer = gold_lookup[qid]['answer']
                # Simple normalization for comparison
                pred_norm = answer.strip().lower()
                gold_norm = gold_answer.strip().lower()
                if pred_norm == gold_norm:
                    exact_matches += 1
        
        if total > 0:
            batch_metrics.append({
                'batch': batch['batch_dir'],
                'total': total,
                'exact_matches': exact_matches,
                'exact_accuracy': exact_matches / total
            })
    
    if not batch_metrics:
        return {}
    
    # Calculate statistics
    accuracies = [bm['exact_accuracy'] for bm in batch_metrics]
    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
    n_batches = len(accuracies)
    
    # Calculate 95% confidence interval
    if n_batches < 2 or std_acc == 0:
        ci_95_lower = ci_95_upper = mean_acc
        ci_95_margin = 0
    else:
        # Use t-distribution for small samples, z for large
        if n_batches > 30:
            z_score = 1.96
        else:
            df = n_batches - 1
            if df <= 10:
                t_score = 2.228
            elif df <= 20:
                t_score = 2.086
            else:
                t_score = 2.042
            z_score = t_score
        
        ci_95_margin = z_score * (std_acc / math.sqrt(n_batches))
        ci_95_lower = max(0.0, mean_acc - ci_95_margin)
        ci_95_upper = min(1.0, mean_acc + ci_95_margin)
    
    return {
        'num_batches': len(batch_metrics),
        'total_questions': sum(bm['total'] for bm in batch_metrics),
        'mean_accuracy': mean_acc,
        'median_accuracy': statistics.median(accuracies),
        'std_accuracy': std_acc,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'ci_95_lower': ci_95_lower,
        'ci_95_upper': ci_95_upper,
        'ci_95_margin': ci_95_margin,
        'batch_details': batch_metrics
    }


def compile_batches(batch_dirs: List[str], pipeline_type: str, 
                    output_dir: str, gold_file: str = None) -> Dict:
    """
    Compile results from multiple batch directories.
    
    Args:
        batch_dirs: List of batch directory paths
        pipeline_type: 'original' or 'experimental'
        output_dir: Output directory for compiled results
        gold_file: Optional gold file for statistics
    
    Returns:
        Compiled results dictionary
    """
    print(f"Compiling {len(batch_dirs)} batches for {pipeline_type} pipeline...")
    
    # Load all batches
    batch_results = []
    for batch_dir in batch_dirs:
        try:
            result = load_batch_result(batch_dir, pipeline_type)
            batch_results.append(result)
            print(f"  ✓ Loaded: {batch_dir}")
        except Exception as e:
            print(f"  ✗ Failed to load {batch_dir}: {e}")
            continue
    
    if not batch_results:
        raise ValueError("No valid batches found!")
    
    # Merge predictions
    print("\nMerging predictions...")
    merged = merge_predictions(batch_results)
    
    # Check for duplicates
    all_qids = set()
    duplicates = []
    for q in merged['tested_questions']:
        qid = q.get('id', '')
        if qid in all_qids:
            duplicates.append(qid)
        all_qids.add(qid)
    
    if duplicates:
        print(f"  ⚠️  Warning: Found {len(duplicates)} duplicate question IDs")
        print(f"     First few: {duplicates[:5]}")
    
    # Calculate batch statistics if gold file provided
    batch_stats = None
    if gold_file and os.path.exists(gold_file):
        print("\nCalculating batch statistics...")
        batch_stats = calculate_batch_statistics(batch_results, gold_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save compiled predictions
    compiled_pred_file = output_path / f"{pipeline_type}_compiled.json"
    with open(compiled_pred_file, 'w') as f:
        json.dump(merged['predictions'], f, indent=2)
    print(f"  ✓ Saved compiled predictions to {compiled_pred_file}")
    
    # Save compiled detailed results
    compiled_detailed = {
        'summary': {
            'total_tested': len(merged['tested_questions']),
            'questions_tested': merged['tested_questions'],
            'pipeline': pipeline_type,
            'num_batches': len(batch_results),
            'batch_dirs': [b['batch_dir'] for b in batch_results]
        },
        'results': [],
        'metadata': merged['metadata']
    }
    
    # Build results list from predictions
    for qid, answer in merged['predictions']['answer'].items():
        compiled_detailed['results'].append({
            'question_id': qid,
            'prediction': answer,
            'supporting_facts': merged['predictions']['sp'].get(qid, [])
        })
    
    compiled_detailed_file = output_path / f"{pipeline_type}_compiled_detailed.json"
    with open(compiled_detailed_file, 'w') as f:
        json.dump(compiled_detailed, f, indent=2)
    print(f"  ✓ Saved compiled detailed results to {compiled_detailed_file}")
    
    # Save batch statistics if available
    if batch_stats:
        stats_file = output_path / f"{pipeline_type}_batch_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        print(f"  ✓ Saved batch statistics to {stats_file}")
        
        print(f"\nBatch Statistics:")
        print(f"  Number of batches: {batch_stats['num_batches']}")
        print(f"  Total questions: {batch_stats['total_questions']}")
        print(f"  Mean accuracy: {batch_stats['mean_accuracy']:.4f}")
        print(f"  Std accuracy: {batch_stats['std_accuracy']:.4f}")
        print(f"  95% CI: [{batch_stats['ci_95_lower']:.4f}, {batch_stats['ci_95_upper']:.4f}]")
        print(f"  Min accuracy: {batch_stats['min_accuracy']:.4f}")
        print(f"  Max accuracy: {batch_stats['max_accuracy']:.4f}")
    
    return {
        'compiled_predictions': str(compiled_pred_file),
        'compiled_detailed': str(compiled_detailed_file),
        'batch_stats': batch_stats,
        'num_batches': len(batch_results),
        'total_questions': len(merged['tested_questions'])
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compile results from multiple batch experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile original pipeline batches
  python compile_batches.py \\
    --batch_dirs exp_2500_random_seed42 exp_2500_random_seed123 exp_2500_random_seed456 \\
    --pipeline_type original \\
    --output_dir compiled_results \\
    --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json
  
  # Compile experimental pipeline batches
  python compile_batches.py \\
    --batch_dirs exp_2500_random_seed42 exp_2500_random_seed123 exp_2500_random_seed456 \\
    --pipeline_type experimental \\
    --output_dir compiled_results \\
    --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json
        """
    )
    
    parser.add_argument('--batch_dirs', type=str, nargs='+', required=True,
                       help='List of batch directory paths (e.g., exp_2500_random_seed42)')
    parser.add_argument('--pipeline_type', type=str, required=True,
                       choices=['original', 'experimental'],
                       help='Pipeline type to compile')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for compiled results')
    parser.add_argument('--gold', type=str, default=None,
                       help='Gold standard file for batch statistics')
    parser.add_argument('--base_dir', type=str, default='../output/hotpot',
                       help='Base directory for batch paths (if relative)')
    
    args = parser.parse_args()
    
    # Resolve batch directories
    base_path = Path(args.base_dir)
    batch_paths = []
    for batch_dir in args.batch_dirs:
        batch_path = Path(batch_dir)
        if not batch_path.is_absolute():
            batch_path = base_path / batch_path
        batch_paths.append(str(batch_path))
    
    # Compile batches
    compile_batches(
        batch_paths,
        args.pipeline_type,
        args.output_dir,
        args.gold
    )
    
    print(f"\n✅ Compilation complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
