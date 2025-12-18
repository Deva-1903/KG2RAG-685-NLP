#!/usr/bin/env python3
"""
Detailed Analysis Script for Extraction and Testing Counts

Analyzes:
1. KG Extraction: How many entities/KGs were extracted
2. Dataset: Total questions in HotpotQA
3. Testing: How many question-answer pairs have been tested
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List


def count_extracted_kgs(kg_dir: str) -> Dict:
    """Count extracted knowledge graphs."""
    kg_path = Path(kg_dir)
    
    if not kg_path.exists():
        return {
            'total_kg_files': 0,
            'kg_files_with_content': 0,
            'total_triplets': 0,
            'entities': []
        }
    
    kg_files = list(kg_path.glob('*.json'))
    total_files = len(kg_files)
    
    entities_with_content = 0
    total_triplets = 0
    entities = []
    
    for kg_file in kg_files:
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
                
            # Count triplets
            file_triplets = 0
            if isinstance(kg_data, dict):
                for seq_key, triplets in kg_data.items():
                    if isinstance(triplets, list):
                        file_triplets += len(triplets)
                
                if file_triplets > 0:
                    entities_with_content += 1
                    total_triplets += file_triplets
                    entities.append({
                        'entity': kg_file.stem.replace('_', '/'),
                        'triplets': file_triplets
                    })
        except Exception as e:
            print(f"  ⚠️  Error reading {kg_file.name}: {e}")
            continue
    
    return {
        'total_kg_files': total_files,
        'kg_files_with_content': entities_with_content,
        'total_triplets': total_triplets,
        'entities': entities
    }


def count_dataset_questions(data_path: str) -> Dict:
    """Count total questions in the dataset."""
    if not os.path.exists(data_path):
        return {
            'total_questions': 0,
            'questions_with_context': 0,
            'unique_entities': 0,
            'entities_per_question': []
        }
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    questions_with_context = 0
    all_entities = set()
    entities_per_q = []
    
    for sample in data:
        if 'context' in sample and len(sample['context']) > 0:
            questions_with_context += 1
            entities_in_q = set()
            for ctx in sample['context']:
                if isinstance(ctx, list) and len(ctx) > 0:
                    entity = ctx[0]
                    entities_in_q.add(entity)
                    all_entities.add(entity)
            entities_per_q.append(len(entities_in_q))
    
    avg_entities = sum(entities_per_q) / len(entities_per_q) if entities_per_q else 0
    
    return {
        'total_questions': total,
        'questions_with_context': questions_with_context,
        'unique_entities': len(all_entities),
        'avg_entities_per_question': avg_entities,
        'min_entities_per_question': min(entities_per_q) if entities_per_q else 0,
        'max_entities_per_question': max(entities_per_q) if entities_per_q else 0
    }


def count_tested_questions(output_dir: str) -> Dict:
    """Count how many questions have been tested from output files."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return {
            'total_tested': 0,
            'experiments': {},
            'unique_questions_tested': set()
        }
    
    all_tested_qids = set()
    experiments = {}
    
    # Look for experiment folders
    for exp_folder in output_path.iterdir():
        if not exp_folder.is_dir() or not exp_folder.name.startswith('exp_'):
            continue
        
        exp_name = exp_folder.name
        tested_in_exp = set()
        
        # Check original and experimental prediction files
        for pred_file in ['original.json', 'experimental.json']:
            pred_path = exp_folder / pred_file
            if pred_path.exists():
                try:
                    with open(pred_path, 'r', encoding='utf-8') as f:
                        pred_data = json.load(f)
                    
                    if 'answer' in pred_data:
                        qids = set(pred_data['answer'].keys())
                        tested_in_exp.update(qids)
                        all_tested_qids.update(qids)
                except Exception as e:
                    print(f"  ⚠️  Error reading {pred_path}: {e}")
        
        # Also check detailed files for tested questions list
        for detailed_file in ['original_detailed.json', 'experimental_detailed.json']:
            detailed_path = exp_folder / detailed_file
            if detailed_path.exists():
                try:
                    with open(detailed_path, 'r', encoding='utf-8') as f:
                        detailed_data = json.load(f)
                    
                    # Check summary for questions_tested
                    if 'summary' in detailed_data and 'questions_tested' in detailed_data['summary']:
                        tested_list = detailed_data['summary']['questions_tested']
                        if isinstance(tested_list, list):
                            for q in tested_list:
                                qid = q.get('id', q) if isinstance(q, dict) else q
                                tested_in_exp.add(str(qid))
                                all_tested_qids.add(str(qid))
                except Exception as e:
                    print(f"  ⚠️  Error reading {detailed_path}: {e}")
        
        if tested_in_exp:
            experiments[exp_name] = {
                'count': len(tested_in_exp),
                'question_ids': list(tested_in_exp)
            }
    
    return {
        'total_tested': len(all_tested_qids),
        'experiments': experiments,
        'unique_questions_tested': all_tested_qids
    }


def analyze_coverage(dataset_info: Dict, kg_info: Dict, tested_info: Dict) -> Dict:
    """Analyze coverage: how many questions have KGs vs tested."""
    # This would require loading the dataset to check which questions have KGs
    # For now, provide basic stats
    return {
        'extraction_coverage': {
            'entities_extracted': kg_info['kg_files_with_content'],
            'total_unique_entities_in_dataset': dataset_info.get('unique_entities', 0),
            'coverage_percentage': (kg_info['kg_files_with_content'] / dataset_info.get('unique_entities', 1) * 100) if dataset_info.get('unique_entities', 0) > 0 else 0
        },
        'testing_coverage': {
            'questions_tested': tested_info['total_tested'],
            'total_questions': dataset_info['total_questions'],
            'coverage_percentage': (tested_info['total_tested'] / dataset_info['total_questions'] * 100) if dataset_info['total_questions'] > 0 else 0
        }
    }


def main():
    """Main analysis function."""
    print("="*80)
    print("DETAILED DATA COUNT ANALYSIS")
    print("="*80)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'hotpotqa' / 'hotpot_dev_distractor_v1.json'
    kg_dir = base_dir / 'data' / 'hotpotqa' / 'kgs' / 'extract_subkgs'
    output_dir = base_dir / 'output' / 'hotpot'
    
    print(f"\n📊 DATASET ANALYSIS")
    print(f"   Dataset file: {data_path}")
    dataset_info = count_dataset_questions(str(data_path))
    print(f"\n   Total Questions in HotpotQA: {dataset_info['total_questions']:,}")
    print(f"   Questions with context: {dataset_info['questions_with_context']:,}")
    print(f"   Unique entities in dataset: {dataset_info['unique_entities']:,}")
    print(f"   Avg entities per question: {dataset_info['avg_entities_per_question']:.2f}")
    print(f"   Min entities per question: {dataset_info['min_entities_per_question']}")
    print(f"   Max entities per question: {dataset_info['max_entities_per_question']}")
    
    print(f"\n🔬 KG EXTRACTION ANALYSIS")
    print(f"   KG directory: {kg_dir}")
    kg_info = count_extracted_kgs(str(kg_dir))
    print(f"\n   Total KG files created: {kg_info['total_kg_files']:,}")
    print(f"   KG files with content: {kg_info['kg_files_with_content']:,}")
    print(f"   Total triplets extracted: {kg_info['total_triplets']:,}")
    if kg_info['entities']:
        avg_triplets = kg_info['total_triplets'] / kg_info['kg_files_with_content']
        print(f"   Avg triplets per entity: {avg_triplets:.1f}")
        top_entities = sorted(kg_info['entities'], key=lambda x: x['triplets'], reverse=True)[:10]
        print(f"\n   Top 10 entities by triplet count:")
        for i, ent in enumerate(top_entities, 1):
            print(f"     {i}. {ent['entity']}: {ent['triplets']} triplets")
    
    print(f"\n🧪 TESTING ANALYSIS")
    print(f"   Output directory: {output_dir}")
    tested_info = count_tested_questions(str(output_dir))
    print(f"\n   Total unique questions tested: {tested_info['total_tested']:,}")
    print(f"   Number of experiments: {len(tested_info['experiments'])}")
    
    if tested_info['experiments']:
        print(f"\n   Breakdown by experiment:")
        for exp_name, exp_data in sorted(tested_info['experiments'].items()):
            print(f"     • {exp_name}: {exp_data['count']:,} questions")
    
    # Coverage analysis
    print(f"\n📈 COVERAGE ANALYSIS")
    coverage = analyze_coverage(dataset_info, kg_info, tested_info)
    print(f"\n   Extraction Coverage:")
    print(f"     Entities extracted: {coverage['extraction_coverage']['entities_extracted']:,}")
    print(f"     Total unique entities: {coverage['extraction_coverage']['total_unique_entities_in_dataset']:,}")
    print(f"     Coverage: {coverage['extraction_coverage']['coverage_percentage']:.2f}%")
    
    print(f"\n   Testing Coverage:")
    print(f"     Questions tested: {coverage['testing_coverage']['questions_tested']:,}")
    print(f"     Total questions: {coverage['testing_coverage']['total_questions']:,}")
    print(f"     Coverage: {coverage['testing_coverage']['coverage_percentage']:.2f}%")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  📦 Extraction: {kg_info['kg_files_with_content']:,} entities extracted from {dataset_info['total_questions']:,} questions")
    print(f"  🧪 Testing: {tested_info['total_tested']:,} questions tested out of {dataset_info['total_questions']:,} total")
    print(f"  📊 Extraction Progress: {coverage['extraction_coverage']['coverage_percentage']:.2f}% of unique entities")
    print(f"  📊 Testing Progress: {coverage['testing_coverage']['coverage_percentage']:.2f}% of total questions")
    print("="*80)
    
    # Save detailed report
    report = {
        'dataset': dataset_info,
        'extraction': kg_info,
        'testing': {
            'total_tested': tested_info['total_tested'],
            'experiments': {k: {'count': v['count']} for k, v in tested_info['experiments'].items()}
        },
        'coverage': coverage
    }
    
    report_path = base_dir / 'data_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Detailed report saved to: {report_path}")


if __name__ == '__main__':
    main()
