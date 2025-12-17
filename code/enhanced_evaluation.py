#!/usr/bin/env python3
"""
Enhanced Evaluation Script for KG²RAG Pipelines

Evaluates predictions with:
- Answer quality (EM, F1, precision, recall)
- Supporting facts quality (EM, F1, precision, recall)
- Joint metrics (answer + supporting facts)
- Token efficiency (accuracy per token)
- Per-question analysis
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

# Import HotpotQA evaluation functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
from hotpot_evaluate_v1 import (
    normalize_answer, f1_score, exact_match_score,
    update_answer, update_sp
)


def load_predictions(prediction_file: str) -> Dict:
    """Load predictions from JSON file."""
    with open(prediction_file, 'r') as f:
        return json.load(f)


def load_gold_data(gold_file: str) -> List[Dict]:
    """Load gold standard data."""
    with open(gold_file, 'r') as f:
        return json.load(f)


def load_metadata(metadata_file: str = None) -> Dict:
    """Load metadata if available (for token counts, etc.)."""
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            # Convert list of metadata to dict by sample_id
            if isinstance(data, list):
                return {item.get('sample_id'): item for item in data}
            elif isinstance(data, dict) and 'metadata' in data:
                metadata_list = data['metadata']
                return {item.get('sample_id'): item for item in metadata_list}
            return data
    return {}


def evaluate_answer_quality(predictions: Dict, gold_data: List[Dict]) -> Dict:
    """Evaluate answer quality metrics."""
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'total': 0
    }
    
    per_question = []
    
    for dp in gold_data:
        cur_id = dp['_id']
        if cur_id not in predictions.get('answer', {}):
            per_question.append({
                'question_id': cur_id,
                'em': 0,
                'f1': 0,
                'prec': 0,
                'recall': 0,
                'prediction': '',
                'ground_truth': dp['answer']
            })
            metrics['total'] += 1
            continue
        
        prediction = predictions['answer'][cur_id]
        gold = dp['answer']
        
        em, prec, recall = update_answer(metrics, prediction, gold)
        f1, _, _ = f1_score(prediction, gold)
        
        per_question.append({
            'question_id': cur_id,
            'em': float(em),
            'f1': f1,
            'prec': prec,
            'recall': recall,
            'prediction': prediction,
            'ground_truth': gold
        })
        metrics['total'] += 1
    
    # Normalize metrics
    if metrics['total'] > 0:
        metrics['em'] /= metrics['total']
        metrics['f1'] /= metrics['total']
        metrics['prec'] /= metrics['total']
        metrics['recall'] /= metrics['total']
    
    return metrics, per_question


def evaluate_supporting_facts(predictions: Dict, gold_data: List[Dict]) -> Dict:
    """Evaluate supporting facts quality."""
    metrics = {
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'total': 0
    }
    
    per_question = []
    
    for dp in gold_data:
        cur_id = dp['_id']
        if cur_id not in predictions.get('sp', {}):
            per_question.append({
                'question_id': cur_id,
                'sp_em': 0,
                'sp_f1': 0,
                'sp_prec': 0,
                'sp_recall': 0,
                'predicted_sp': [],
                'gold_sp': dp['supporting_facts']
            })
            metrics['total'] += 1
            continue
        
        prediction_sp = predictions['sp'][cur_id]
        gold_sp = dp['supporting_facts']
        
        sp_em, sp_prec, sp_recall = update_sp(metrics, prediction_sp, gold_sp)
        
        # Calculate F1
        pred_set = set(map(tuple, prediction_sp))
        gold_set = set(map(tuple, gold_sp))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        sp_f1 = 2 * sp_prec * sp_recall / (sp_prec + sp_recall) if (sp_prec + sp_recall) > 0 else 0.0
        
        per_question.append({
            'question_id': cur_id,
            'sp_em': float(sp_em),
            'sp_f1': sp_f1,
            'sp_prec': sp_prec,
            'sp_recall': sp_recall,
            'predicted_sp': prediction_sp,
            'gold_sp': gold_sp
        })
        metrics['total'] += 1
    
    # Normalize metrics
    if metrics['total'] > 0:
        metrics['sp_em'] /= metrics['total']
        metrics['sp_f1'] /= metrics['total']
        metrics['sp_prec'] /= metrics['total']
        metrics['sp_recall'] /= metrics['total']
    
    return metrics, per_question


def evaluate_joint_metrics(answer_metrics: Dict, sp_metrics: Dict, 
                          answer_per_q: List[Dict], sp_per_q: List[Dict]) -> Dict:
    """Evaluate joint metrics (answer + supporting facts)."""
    # Aggregate joint metrics
    joint_metrics = {
        'joint_em': 0,
        'joint_f1': 0,
        'joint_prec': 0,
        'joint_recall': 0,
        'total': 0
    }
    
    # Create lookup for per-question metrics
    answer_lookup = {q['question_id']: q for q in answer_per_q}
    sp_lookup = {q['question_id']: q for q in sp_per_q}
    
    per_question = []
    
    for qid in answer_lookup.keys():
        if qid not in sp_lookup:
            continue
        
        aq = answer_lookup[qid]
        sq = sp_lookup[qid]
        
        # Joint EM: both answer and SP must be exact match
        joint_em = aq['em'] * sq['sp_em']
        
        # Joint precision and recall
        joint_prec = aq['prec'] * sq['sp_prec']
        joint_recall = aq['recall'] * sq['sp_recall']
        
        # Joint F1
        if joint_prec + joint_recall > 0:
            joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        else:
            joint_f1 = 0.0
        
        joint_metrics['joint_em'] += joint_em
        joint_metrics['joint_f1'] += joint_f1
        joint_metrics['joint_prec'] += joint_prec
        joint_metrics['joint_recall'] += joint_recall
        joint_metrics['total'] += 1
        
        per_question.append({
            'question_id': qid,
            'joint_em': joint_em,
            'joint_f1': joint_f1,
            'joint_prec': joint_prec,
            'joint_recall': joint_recall
        })
    
    # Normalize
    if joint_metrics['total'] > 0:
        joint_metrics['joint_em'] /= joint_metrics['total']
        joint_metrics['joint_f1'] /= joint_metrics['total']
        joint_metrics['joint_prec'] /= joint_metrics['total']
        joint_metrics['joint_recall'] /= joint_metrics['total']
    
    return joint_metrics, per_question


def calculate_token_efficiency(metadata: Dict, answer_per_q: List[Dict]) -> Dict:
    """Calculate token efficiency metrics."""
    token_counts = []
    correct_with_tokens = []
    
    for q in answer_per_q:
        qid = q['question_id']
        if qid in metadata:
            tokens = metadata[qid].get('tokens_used', 0)
            if tokens > 0:
                token_counts.append(tokens)
                # Accuracy per token
                if q['em'] > 0:
                    correct_with_tokens.append(tokens)
    
    if not token_counts:
        return {
            'avg_tokens': 0,
            'median_tokens': 0,
            'min_tokens': 0,
            'max_tokens': 0,
            'accuracy_per_1k_tokens': 0,
            'total_questions_with_tokens': 0
        }
    
    # Calculate accuracy per 1000 tokens
    total_tokens = sum(token_counts)
    total_correct = len(correct_with_tokens)
    accuracy_per_1k = (total_correct / len(token_counts)) * 1000 / (total_tokens / len(token_counts)) if token_counts else 0
    
    return {
        'avg_tokens': statistics.mean(token_counts),
        'median_tokens': statistics.median(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'accuracy_per_1k_tokens': accuracy_per_1k,
        'total_questions_with_tokens': len(token_counts)
    }


def analyze_failures(answer_per_q: List[Dict], sp_per_q: List[Dict], 
                    gold_data: List[Dict]) -> Dict:
    """Analyze where the pipeline fails."""
    failures = {
        'answer_only_failures': [],  # SP correct but answer wrong
        'sp_only_failures': [],      # Answer correct but SP wrong
        'both_failures': [],          # Both wrong
        'both_correct': []            # Both correct
    }
    
    answer_lookup = {q['question_id']: q for q in answer_per_q}
    sp_lookup = {q['question_id']: q for q in sp_per_q}
    gold_lookup = {dp['_id']: dp for dp in gold_data}
    
    for qid in answer_lookup.keys():
        if qid not in sp_lookup or qid not in gold_lookup:
            continue
        
        aq = answer_lookup[qid]
        sq = sp_lookup[qid]
        gold = gold_lookup[qid]
        
        answer_correct = aq['em'] > 0
        sp_correct = sq['sp_em'] > 0
        
        failure_info = {
            'question_id': qid,
            'question': gold.get('question', ''),
            'answer_correct': answer_correct,
            'sp_correct': sp_correct,
            'answer_em': aq['em'],
            'answer_f1': aq['f1'],
            'sp_em': sq['sp_em'],
            'sp_f1': sq['sp_f1']
        }
        
        if answer_correct and sp_correct:
            failures['both_correct'].append(failure_info)
        elif answer_correct and not sp_correct:
            failures['sp_only_failures'].append(failure_info)
        elif not answer_correct and sp_correct:
            failures['answer_only_failures'].append(failure_info)
        else:
            failures['both_failures'].append(failure_info)
    
    return failures


def comprehensive_evaluate(prediction_file: str, gold_file: str, 
                          metadata_file: str = None,
                          output_file: str = None) -> Dict:
    """Comprehensive evaluation of predictions."""
    
    print(f"Loading predictions from {prediction_file}...")
    predictions = load_predictions(prediction_file)
    
    print(f"Loading gold data from {gold_file}...")
    gold_data = load_gold_data(gold_file)
    
    metadata = {}
    if metadata_file:
        print(f"Loading metadata from {metadata_file}...")
        metadata = load_metadata(metadata_file)
    
    print("Evaluating answer quality...")
    answer_metrics, answer_per_q = evaluate_answer_quality(predictions, gold_data)
    
    print("Evaluating supporting facts...")
    sp_metrics, sp_per_q = evaluate_supporting_facts(predictions, gold_data)
    
    print("Evaluating joint metrics...")
    joint_metrics, joint_per_q = evaluate_joint_metrics(
        answer_metrics, sp_metrics, answer_per_q, sp_per_q
    )
    
    print("Calculating token efficiency...")
    token_efficiency = calculate_token_efficiency(metadata, answer_per_q)
    
    print("Analyzing failures...")
    failures = analyze_failures(answer_per_q, sp_per_q, gold_data)
    
    # Compile results
    results = {
        'answer_metrics': answer_metrics,
        'supporting_facts_metrics': sp_metrics,
        'joint_metrics': joint_metrics,
        'token_efficiency': token_efficiency,
        'failure_analysis': {
            'answer_only_failures': len(failures['answer_only_failures']),
            'sp_only_failures': len(failures['sp_only_failures']),
            'both_failures': len(failures['both_failures']),
            'both_correct': len(failures['both_correct']),
            'total': len(failures['both_correct']) + len(failures['answer_only_failures']) + 
                     len(failures['sp_only_failures']) + len(failures['both_failures'])
        },
        'per_question_analysis': {
            'answer': answer_per_q,
            'supporting_facts': sp_per_q,
            'joint': joint_per_q
        },
        'detailed_failures': failures
    }
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    print(f"\nAnswer Quality:")
    print(f"  EM: {answer_metrics['em']:.4f}")
    print(f"  F1: {answer_metrics['f1']:.4f}")
    print(f"  Precision: {answer_metrics['prec']:.4f}")
    print(f"  Recall: {answer_metrics['recall']:.4f}")
    
    print(f"\nSupporting Facts Quality:")
    print(f"  EM: {sp_metrics['sp_em']:.4f}")
    print(f"  F1: {sp_metrics['sp_f1']:.4f}")
    print(f"  Precision: {sp_metrics['sp_prec']:.4f}")
    print(f"  Recall: {sp_metrics['sp_recall']:.4f}")
    
    print(f"\nJoint Metrics (Answer + Supporting Facts):")
    print(f"  EM: {joint_metrics['joint_em']:.4f}")
    print(f"  F1: {joint_metrics['joint_f1']:.4f}")
    print(f"  Precision: {joint_metrics['joint_prec']:.4f}")
    print(f"  Recall: {joint_metrics['joint_recall']:.4f}")
    
    if token_efficiency['total_questions_with_tokens'] > 0:
        print(f"\nToken Efficiency:")
        print(f"  Avg tokens per question: {token_efficiency['avg_tokens']:.1f}")
        print(f"  Median tokens: {token_efficiency['median_tokens']:.1f}")
        print(f"  Accuracy per 1k tokens: {token_efficiency['accuracy_per_1k_tokens']:.4f}")
    
    print(f"\nFailure Analysis:")
    print(f"  Both correct: {failures['both_correct']}")
    print(f"  Answer only failures: {len(failures['answer_only_failures'])}")
    print(f"  SP only failures: {len(failures['sp_only_failures'])}")
    print(f"  Both failures: {len(failures['both_failures'])}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced evaluation for KG²RAG pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions
  python enhanced_evaluation.py \\
    --prediction original.json \\
    --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json \\
    --metadata original_detailed.json \\
    --output evaluation_results.json
        """
    )
    
    parser.add_argument('--prediction', type=str, required=True,
                       help='Path to prediction JSON file')
    parser.add_argument('--gold', type=str, required=True,
                       help='Path to gold standard JSON file')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata file (for token counts)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    comprehensive_evaluate(
        args.prediction,
        args.gold,
        args.metadata,
        args.output
    )


if __name__ == '__main__':
    main()
