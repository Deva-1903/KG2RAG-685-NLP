#!/usr/bin/env python3
"""
Split HotpotQA dataset into multiple files for easier processing.
"""

import ujson as json
import os
import math

def split_dataset(input_path, output_dir, num_splits=3):
    """
    Split dataset into multiple files.
    
    Args:
        input_path: Path to original dataset JSON file
        output_dir: Directory to save split files
        num_splits: Number of files to split into
    """
    print(f"=" * 80)
    print(f"Splitting dataset into {num_splits} files...")
    print(f"=" * 80)
    
    # Load original dataset
    print(f"Loading dataset from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    total_questions = len(data)
    print(f"Total questions: {total_questions}")
    
    # Calculate split sizes
    questions_per_split = math.ceil(total_questions / num_splits)
    print(f"Questions per split: ~{questions_per_split}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and save
    for i in range(num_splits):
        start_idx = i * questions_per_split
        end_idx = min((i + 1) * questions_per_split, total_questions)
        
        split_data = data[start_idx:end_idx]
        output_path = os.path.join(output_dir, f'hotpot_dev_distractor_v1_part{i+1}.json')
        
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        
        print(f"Part {i+1}: Questions {start_idx}-{end_idx-1} ({len(split_data)} questions) → {output_path}")
    
    print(f"\n✓ Split complete! Created {num_splits} files in {output_dir}")
    print(f"\nTo process each part, modify hotpot_extraction.py to use:")
    for i in range(num_splits):
        print(f"  Part {i+1}: data_path = '{output_dir}/hotpot_dev_distractor_v1_part{i+1}.json'")

if __name__ == '__main__':
    # Default paths
    input_path = '../../data/hotpotqa/hotpot_dev_distractor_v1.json'
    output_dir = '../../data/hotpotqa/splits'
    num_splits = 5  # Split into 5 parts for team distribution
    
    split_dataset(input_path, output_dir, num_splits)
