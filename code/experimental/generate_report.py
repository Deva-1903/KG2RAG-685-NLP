#!/usr/bin/env python3
"""
Generate Comprehensive Report from Multiple Experiment Results

Analyzes multiple experiment JSON files and generates a comprehensive report
comparing original vs experimental KG²RAG across different configurations.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_experiment_result(detailed_json_path: str) -> Dict:
    """Load and parse experiment result from detailed JSON."""
    with open(detailed_json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    summary = data.get('summary', {})
    
    # Calculate metrics
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
        'questions_tested': summary.get('questions_tested', []),
    }


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    import re
    if not text:
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def is_yes_no_question(question: str) -> bool:
    """Check if question is yes/no type."""
    question_lower = question.lower().strip()
    return (question_lower.startswith('are ') or 
            question_lower.startswith('is ') or 
            question_lower.startswith('were ') or 
            question_lower.startswith('was ') or
            question_lower.startswith('do ') or
            question_lower.startswith('does '))


def parse_folder_name(folder_name: str) -> Dict:
    """Parse experiment configuration from folder name."""
    # Format: exp_200_random_seed42 or exp_250_first
    
    parts = folder_name.split('_')
    
    config = {
        'num_questions': None,
        'sampling': None,
        'seed': None,
    }
    
    # Find number of questions (should be second part after 'exp')
    if len(parts) >= 2 and parts[0] == 'exp':
        try:
            config['num_questions'] = int(parts[1])
        except ValueError:
            pass
    
    # Find sampling method
    if 'random' in parts:
        config['sampling'] = 'random'
        # Find seed
        for part in parts:
            if part.startswith('seed'):
                try:
                    config['seed'] = int(part.replace('seed', ''))
                except ValueError:
                    pass
    elif 'first' in parts:
        config['sampling'] = 'first'
    
    return config


def find_experiment_files(output_dir: str) -> Dict[Tuple, Dict]:
    """
    Find all experiment folders and group by configuration.
    
    Returns:
        Dict mapping (num_questions, sampling, seed) -> {
            'original': path,
            'experimental': path,
            'folder': path
        }
    """
    experiments = defaultdict(dict)
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return {}
    
    # Check if the output_dir itself is an experiment folder
    if output_path.is_dir() and output_path.name.startswith('exp_'):
        # Process this folder directly
        folder_name = output_path.name
        config = parse_folder_name(folder_name)
        
        if config['num_questions'] is not None:
            key = (
                config['num_questions'],
                config['sampling'],
                config['seed']
            )
            
            original_detailed = output_path / "original_detailed.json"
            experimental_detailed = output_path / "experimental_detailed.json"
            
            if original_detailed.exists():
                experiments[key]['original'] = str(original_detailed)
            if experimental_detailed.exists():
                experiments[key]['experimental'] = str(experimental_detailed)
            
            if 'original' in experiments[key] or 'experimental' in experiments[key]:
                experiments[key]['folder'] = str(output_path)
        
        return dict(experiments)
    
    # Otherwise, find all experiment folders (starting with 'exp_')
    for item_path in output_path.iterdir():
        if not item_path.is_dir():
            continue  # Skip files, only process folders
        
        folder_name = item_path.name
        if not folder_name.startswith('exp_'):
            continue  # Skip non-experiment folders
        
        # Parse folder name to get configuration
        config = parse_folder_name(folder_name)
        
        if config['num_questions'] is None:
            continue  # Skip if couldn't parse
        
        key = (
            config['num_questions'],
            config['sampling'],
            config['seed']
        )
        
        # Look for detailed JSON files in the folder
        original_detailed = item_path / "original_detailed.json"
        experimental_detailed = item_path / "experimental_detailed.json"
        
        if original_detailed.exists():
            experiments[key]['original'] = str(original_detailed)
        if experimental_detailed.exists():
            experiments[key]['experimental'] = str(experimental_detailed)
        
        if 'original' in experiments[key] or 'experimental' in experiments[key]:
            experiments[key]['folder'] = str(item_path)
    
    return dict(experiments)


def generate_report(experiments: Dict, output_file: str = None):
    """Generate comprehensive report from experiments."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("KG²RAG EXPERIMENTAL PIPELINE - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Sort experiments by num_questions, then sampling
    sorted_experiments = sorted(
        experiments.items(),
        key=lambda x: (x[0][0], x[0][1] or '', x[0][2] or 0)
    )
    
    # Summary table
    report_lines.append("SUMMARY TABLE")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Config':<25} {'Original':<20} {'Experimental':<20} {'Improvement':<15}")
    report_lines.append("-" * 80)
    
    all_results = []
    
    for (num_q, sampling, seed), files in sorted_experiments:
        # Check if both files exist
        if 'original' not in files or 'experimental' not in files:
            if 'folder' in files:
                print(f"⚠️  Incomplete experiment in {files['folder']}: missing {'original' if 'original' not in files else 'experimental'} results")
            continue
        
        # Load results
        original_metrics = load_experiment_result(files['original'])
        experimental_metrics = load_experiment_result(files['experimental'])
        
        # Format config string
        if sampling == 'random':
            config_str = f"{num_q}Q, Random (seed={seed})"
        else:
            config_str = f"{num_q}Q, First N"
        
        # Calculate improvement
        improvement = experimental_metrics['overall_accuracy'] - original_metrics['overall_accuracy']
        improvement_str = f"{improvement:+.1f}%"
        
        report_lines.append(
            f"{config_str:<25} "
            f"{original_metrics['overall_accuracy']:>6.1f}% ({original_metrics['exact_accuracy']:>5.1f}% exact)  "
            f"{experimental_metrics['overall_accuracy']:>6.1f}% ({experimental_metrics['exact_accuracy']:>5.1f}% exact)  "
            f"{improvement_str:>15}"
        )
        
        all_results.append({
            'config': config_str,
            'num_questions': num_q,
            'sampling': sampling,
            'seed': seed,
            'folder': files.get('folder', ''),
            'original': original_metrics,
            'experimental': experimental_metrics,
            'improvement': improvement,
        })
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DETAILED RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Detailed results for each experiment
    for result in all_results:
        report_lines.append(f"Configuration: {result['config']}")
        if result.get('folder'):
            report_lines.append(f"Experiment folder: {result['folder']}")
        report_lines.append("-" * 80)
        
        orig = result['original']
        exp = result['experimental']
        
        report_lines.append(f"Original KG²RAG:")
        report_lines.append(f"  Total Questions: {orig['total']}")
        report_lines.append(f"  Overall Accuracy: {orig['overall_accuracy']:.1f}%")
        report_lines.append(f"  Exact Matches: {orig['exact_matches']} ({orig['exact_accuracy']:.1f}%)")
        report_lines.append(f"  Partial Matches: {orig['partial_matches']} ({orig['partial_accuracy']:.1f}%)")
        report_lines.append(f"  Yes/No Accuracy: {orig['yes_no_accuracy']:.1f}% ({orig['yes_no_questions']} questions)")
        report_lines.append(f"  Factoid Accuracy: {orig['factoid_accuracy']:.1f}% ({orig['factoid_questions']} questions)")
        report_lines.append("")
        
        report_lines.append(f"Experimental KG²RAG:")
        report_lines.append(f"  Total Questions: {exp['total']}")
        report_lines.append(f"  Overall Accuracy: {exp['overall_accuracy']:.1f}%")
        report_lines.append(f"  Exact Matches: {exp['exact_matches']} ({exp['exact_accuracy']:.1f}%)")
        report_lines.append(f"  Partial Matches: {exp['partial_matches']} ({exp['partial_accuracy']:.1f}%)")
        report_lines.append(f"  Yes/No Accuracy: {exp['yes_no_accuracy']:.1f}% ({exp['yes_no_questions']} questions)")
        report_lines.append(f"  Factoid Accuracy: {exp['factoid_accuracy']:.1f}% ({exp['factoid_questions']} questions)")
        report_lines.append("")
        
        report_lines.append(f"Improvement Analysis:")
        report_lines.append(f"  Overall Accuracy: {result['improvement']:+.1f}%")
        report_lines.append(f"  Exact Match: {exp['exact_accuracy'] - orig['exact_accuracy']:+.1f}%")
        report_lines.append(f"  Factoid Accuracy: {exp['factoid_accuracy'] - orig['factoid_accuracy']:+.1f}%")
        if orig['yes_no_questions'] > 0 and exp['yes_no_questions'] > 0:
            report_lines.append(f"  Yes/No Accuracy: {exp['yes_no_accuracy'] - orig['yes_no_accuracy']:+.1f}%")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Overall statistics
    if all_results:
        avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
        max_improvement = max(r['improvement'] for r in all_results)
        min_improvement = min(r['improvement'] for r in all_results)
        
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of experiments: {len(all_results)}")
        report_lines.append(f"Average improvement: {avg_improvement:+.1f}%")
        report_lines.append(f"Maximum improvement: {max_improvement:+.1f}%")
        report_lines.append(f"Minimum improvement: {min_improvement:+.1f}%")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\n✅ Report saved to: {output_file}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive report from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from all experiments in output directory
  python generate_report.py --output_dir ../output/hotpot
  
  # Generate report and save to file
  python generate_report.py --output_dir ../output/hotpot --report_file report.md
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output/hotpot',
        help='Directory containing experiment results (default: ../output/hotpot). Looks for folders starting with "exp_"'
    )
    
    parser.add_argument(
        '--report_file',
        type=str,
        default=None,
        help='Output file for report (default: print to console)'
    )
    
    args = parser.parse_args()
    
    # Find all experiment files
    experiments = find_experiment_files(args.output_dir)
    
    if not experiments:
        print(f"❌ No experiment results found in {args.output_dir}")
        print("   Make sure you've run experiments using run_experiments.py")
        print("   Looking for folders starting with 'exp_' (e.g., exp_200_random_seed42/)")
        print("   Each experiment should be in its own folder with original_detailed.json and experimental_detailed.json")
        return
    
    print(f"Found {len(experiments)} experiment configurations")
    
    # Generate report
    generate_report(experiments, args.report_file)


if __name__ == '__main__':
    main()
