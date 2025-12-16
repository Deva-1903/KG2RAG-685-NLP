#!/usr/bin/env python3
"""
Compare Original KG²RAG vs Experimental KG²RAG Results

Compares results from:
- Original: kg_rag_distractor_100.py output
- Experimental: kg_rag_enhanced_100.py output (multi-view + knapsack)
"""

import json
import re
from typing import Dict, List
import sys


def normalize(text):
    """Normalize text for comparison."""
    if not text:
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def is_yes_no_question(question):
    """Check if question is yes/no type."""
    question_lower = question.lower().strip()
    return (question_lower.startswith('are ') or 
            question_lower.startswith('is ') or 
            question_lower.startswith('were ') or 
            question_lower.startswith('was ') or
            question_lower.startswith('do ') or
            question_lower.startswith('does '))


def analyze_results(file_path, label):
    """Analyze a results file and return metrics."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    total = len(results)
    
    exact_matches = 0
    partial_matches = 0
    no_matches = 0
    empty_predictions = 0
    
    yes_no_questions = 0
    yes_no_correct = 0
    factoid_questions = 0
    factoid_correct = 0
    
    for r in results:
        pred = r.get('prediction', '').strip()
        truth = r.get('ground_truth', '').strip()
        question = r.get('question', '')
        
        if not pred:
            empty_predictions += 1
            no_matches += 1
            continue
        
        pred_norm = normalize(pred)
        truth_norm = normalize(truth)
        
        is_yes_no = is_yes_no_question(question)
        if is_yes_no:
            yes_no_questions += 1
        else:
            factoid_questions += 1
        
        if pred_norm == truth_norm:
            exact_matches += 1
            if is_yes_no:
                yes_no_correct += 1
            else:
                factoid_correct += 1
        elif truth_norm in pred_norm or pred_norm in truth_norm:
            partial_matches += 1
            if is_yes_no:
                yes_no_correct += 1
            else:
                factoid_correct += 1
        else:
            no_matches += 1
    
    overall_correct = exact_matches + partial_matches
    
    return {
        'label': label,
        'total': total,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'no_matches': no_matches,
        'empty_predictions': empty_predictions,
        'overall_correct': overall_correct,
        'exact_accuracy': exact_matches / total * 100 if total > 0 else 0,
        'partial_accuracy': partial_matches / total * 100 if total > 0 else 0,
        'overall_accuracy': overall_correct / total * 100 if total > 0 else 0,
        'yes_no_questions': yes_no_questions,
        'yes_no_correct': yes_no_correct,
        'yes_no_accuracy': yes_no_correct / yes_no_questions * 100 if yes_no_questions > 0 else 0,
        'factoid_questions': factoid_questions,
        'factoid_correct': factoid_correct,
        'factoid_accuracy': factoid_correct / factoid_questions * 100 if factoid_questions > 0 else 0,
    }


def compare_results(original_file, enhanced_file):
    """Compare original vs enhanced results."""
    
    print("=" * 80)
    print("Original KG²RAG vs Experimental KG²RAG Comparison")
    print("=" * 80)
    print()
    
    # Analyze both files
    original_results = analyze_results(original_file, "Original KG²RAG")
    enhanced_results = analyze_results(enhanced_file, "Experimental KG²RAG")
    
    # Print individual results
    print(f"{original_results['label']} Results:")
    print("-" * 80)
    print(f"  Total Questions: {original_results['total']}")
    print(f"  Exact Matches: {original_results['exact_matches']} ({original_results['exact_accuracy']:.1f}%)")
    print(f"  Partial Matches: {original_results['partial_matches']} ({original_results['partial_accuracy']:.1f}%)")
    print(f"  Overall Accuracy: {original_results['overall_accuracy']:.1f}%")
    print(f"  Yes/No Questions: {original_results['yes_no_questions']} (Accuracy: {original_results['yes_no_accuracy']:.1f}%)")
    print(f"  Factoid Questions: {original_results['factoid_questions']} (Accuracy: {original_results['factoid_accuracy']:.1f}%)")
    print()
    
    print(f"{enhanced_results['label']} Results:")
    print("-" * 80)
    print(f"  Total Questions: {enhanced_results['total']}")
    print(f"  Exact Matches: {enhanced_results['exact_matches']} ({enhanced_results['exact_accuracy']:.1f}%)")
    print(f"  Partial Matches: {enhanced_results['partial_matches']} ({enhanced_results['partial_accuracy']:.1f}%)")
    print(f"  Overall Accuracy: {enhanced_results['overall_accuracy']:.1f}%")
    print(f"  Yes/No Questions: {enhanced_results['yes_no_questions']} (Accuracy: {enhanced_results['yes_no_accuracy']:.1f}%)")
    print(f"  Factoid Questions: {enhanced_results['factoid_questions']} (Accuracy: {enhanced_results['factoid_accuracy']:.1f}%)")
    print()
    
    # Comparison
    print("Improvement Analysis:")
    print("-" * 80)
    accuracy_diff = enhanced_results['overall_accuracy'] - original_results['overall_accuracy']
    exact_diff = enhanced_results['exact_accuracy'] - original_results['exact_accuracy']
    factoid_diff = enhanced_results['factoid_accuracy'] - original_results['factoid_accuracy']
    yes_no_diff = enhanced_results['yes_no_accuracy'] - original_results['yes_no_accuracy']
    
    print(f"  Overall Accuracy: {accuracy_diff:+.1f}% ({'✓ Improvement' if accuracy_diff > 0 else '✗ Degradation' if accuracy_diff < 0 else '≈ Same'})")
    print(f"  Exact Match: {exact_diff:+.1f}% ({'✓ Improvement' if exact_diff > 0 else '✗ Degradation' if exact_diff < 0 else '≈ Same'})")
    print(f"  Factoid Accuracy: {factoid_diff:+.1f}% ({'✓ Improvement' if factoid_diff > 0 else '✗ Degradation' if factoid_diff < 0 else '≈ Same'})")
    if original_results['yes_no_questions'] > 0 and enhanced_results['yes_no_questions'] > 0:
        print(f"  Yes/No Accuracy: {yes_no_diff:+.1f}% ({'✓ Improvement' if yes_no_diff > 0 else '✗ Degradation' if yes_no_diff < 0 else '≈ Same'})")
    print()
    
    # Summary
    print("Summary:")
    print("-" * 80)
    if accuracy_diff > 2:
        print(f"  ✓ Experimental pipeline performs significantly better (+{accuracy_diff:.1f}%)")
    elif accuracy_diff < -2:
        print(f"  ✗ Experimental pipeline performs worse ({accuracy_diff:.1f}%)")
    else:
        print(f"  ≈ Both pipelines perform similarly (difference: {accuracy_diff:.1f}%)")
    print()
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_original_vs_enhanced.py <original_detailed.json> <experimental_detailed.json>")
        print("\nExample:")
        print("  python compare_original_vs_enhanced.py \\")
        print("    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \\")
        print("    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_experimental_100_detailed.json")
        sys.exit(1)
    
    original_file = sys.argv[1]
    enhanced_file = sys.argv[2]
    
    compare_results(original_file, enhanced_file)
