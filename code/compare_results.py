#!/usr/bin/env python3
"""
Compare results from two different test runs (first 100 vs random 100)
"""

import json
import re
from collections import Counter

def normalize(text):
    """Normalize text for comparison"""
    if not text:
        return ''
    text = str(text).lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def is_yes_no_question(question):
    """Check if question is yes/no type"""
    question_lower = question.lower().strip()
    return (question_lower.startswith('are ') or 
            question_lower.startswith('is ') or 
            question_lower.startswith('were ') or 
            question_lower.startswith('was ') or
            question_lower.startswith('do ') or
            question_lower.startswith('does '))

def analyze_results(file_path, label):
    """Analyze a results file and return metrics"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    total = len(results)
    
    # Metrics
    exact_matches = 0
    partial_matches = 0
    no_matches = 0
    empty_predictions = 0
    has_supporting_facts = 0
    total_supporting_facts = 0
    
    # Answer length stats
    answer_lengths = []
    prediction_lengths = []
    
    # Question types
    yes_no_questions = 0
    yes_no_correct = 0
    factoid_questions = 0
    factoid_correct = 0
    
    # Question IDs for overlap analysis
    question_ids = set()
    
    for r in results:
        pred = r.get('prediction', '').strip()
        truth = r.get('ground_truth', '').strip()
        question = r.get('question', '')
        question_id = r.get('question_id', '')
        sps = r.get('supporting_facts', [])
        
        question_ids.add(question_id)
        
        # Check if empty
        if not pred:
            empty_predictions += 1
            no_matches += 1
            continue
        
        # Supporting facts
        if sps and len(sps) > 0:
            has_supporting_facts += 1
            total_supporting_facts += len(sps)
        
        # Answer length
        answer_lengths.append(len(truth))
        prediction_lengths.append(len(pred))
        
        # Normalize for comparison
        pred_norm = normalize(pred)
        truth_norm = normalize(truth)
        
        # Check question type
        is_yes_no = is_yes_no_question(question)
        
        if is_yes_no:
            yes_no_questions += 1
        else:
            factoid_questions += 1
        
        # Check matches
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
        'has_supporting_facts': has_supporting_facts,
        'avg_supporting_facts': total_supporting_facts / has_supporting_facts if has_supporting_facts > 0 else 0,
        'yes_no_questions': yes_no_questions,
        'yes_no_correct': yes_no_correct,
        'yes_no_accuracy': yes_no_correct / yes_no_questions * 100 if yes_no_questions > 0 else 0,
        'factoid_questions': factoid_questions,
        'factoid_correct': factoid_correct,
        'factoid_accuracy': factoid_correct / factoid_questions * 100 if factoid_questions > 0 else 0,
        'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        'avg_prediction_length': sum(prediction_lengths) / len(prediction_lengths) if prediction_lengths else 0,
        'question_ids': question_ids
    }

def compare_results(file1, file2, label1="First 100", label2="Random 100"):
    """Compare two result files"""
    
    print("=" * 80)
    print("KG²RAG Results Comparison")
    print("=" * 80)
    print()
    
    # Analyze both files
    results1 = analyze_results(file1, label1)
    results2 = analyze_results(file2, label2)
    
    # Print individual results
    print(f"{label1} Results:")
    print("-" * 80)
    print(f"  Total Questions: {results1['total']}")
    print(f"  Exact Matches: {results1['exact_matches']} ({results1['exact_accuracy']:.1f}%)")
    print(f"  Partial Matches: {results1['partial_matches']} ({results1['partial_accuracy']:.1f}%)")
    print(f"  Overall Accuracy: {results1['overall_accuracy']:.1f}%")
    print(f"  Yes/No Questions: {results1['yes_no_questions']} (Accuracy: {results1['yes_no_accuracy']:.1f}%)")
    print(f"  Factoid Questions: {results1['factoid_questions']} (Accuracy: {results1['factoid_accuracy']:.1f}%)")
    print(f"  Avg Supporting Facts: {results1['avg_supporting_facts']:.1f}")
    print()
    
    print(f"{label2} Results:")
    print("-" * 80)
    print(f"  Total Questions: {results2['total']}")
    print(f"  Exact Matches: {results2['exact_matches']} ({results2['exact_accuracy']:.1f}%)")
    print(f"  Partial Matches: {results2['partial_matches']} ({results2['partial_accuracy']:.1f}%)")
    print(f"  Overall Accuracy: {results2['overall_accuracy']:.1f}%")
    print(f"  Yes/No Questions: {results2['yes_no_questions']} (Accuracy: {results2['yes_no_accuracy']:.1f}%)")
    print(f"  Factoid Questions: {results2['factoid_questions']} (Accuracy: {results2['factoid_accuracy']:.1f}%)")
    print(f"  Avg Supporting Facts: {results2['avg_supporting_facts']:.1f}")
    print()
    
    # Comparison
    print("Comparison:")
    print("-" * 80)
    accuracy_diff = results2['overall_accuracy'] - results1['overall_accuracy']
    print(f"  Overall Accuracy Difference: {accuracy_diff:+.1f}% ({label2} - {label1})")
    
    exact_diff = results2['exact_accuracy'] - results1['exact_accuracy']
    print(f"  Exact Match Difference: {exact_diff:+.1f}% ({label2} - {label1})")
    
    yes_no_diff = results2['yes_no_accuracy'] - results1['yes_no_accuracy']
    print(f"  Yes/No Accuracy Difference: {yes_no_diff:+.1f}% ({label2} - {label1})")
    
    factoid_diff = results2['factoid_accuracy'] - results1['factoid_accuracy']
    print(f"  Factoid Accuracy Difference: {factoid_diff:+.1f}% ({label2} - {label1})")
    
    # Question overlap
    overlap = results1['question_ids'] & results2['question_ids']
    print(f"  Question Overlap: {len(overlap)} questions appear in both sets")
    print()
    
    # Summary
    print("Summary:")
    print("-" * 80)
    if accuracy_diff > 2:
        print(f"  ✓ {label2} performs significantly better ({accuracy_diff:.1f}% higher)")
    elif accuracy_diff < -2:
        print(f"  ✗ {label2} performs significantly worse ({abs(accuracy_diff):.1f}% lower)")
    else:
        print(f"  ≈ Both sets perform similarly (difference: {accuracy_diff:.1f}%)")
    print()
    print("=" * 80)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <file1_detailed.json> <file2_detailed.json> [label1] [label2]")
        print("\nExample:")
        print("  python compare_results.py ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \\")
        print("                            ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random_detailed.json")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    label1 = sys.argv[3] if len(sys.argv) > 3 else "First 100"
    label2 = sys.argv[4] if len(sys.argv) > 4 else "Random 100"
    
    compare_results(file1, file2, label1, label2)
