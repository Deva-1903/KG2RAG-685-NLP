#!/usr/bin/env python3
"""
Verify that split files contain the same data as the original dataset.
Checks for:
- Total question count matches
- No missing questions
- No duplicate questions
- All questions are present
"""

import ujson as json
import os
from collections import Counter

def verify_splits(original_path, splits_dir, num_splits=5):
    """
    Verify that split files match the original dataset.
    
    Args:
        original_path: Path to original dataset JSON file
        splits_dir: Directory containing split files
        num_splits: Number of split files expected
    """
    print("=" * 80)
    print("Verifying Split Files")
    print("=" * 80)
    print()
    
    # Load original dataset
    print(f"Loading original dataset: {original_path}")
    if not os.path.exists(original_path):
        print(f"✗ Error: Original file not found: {original_path}")
        return False
    
    with open(original_path) as f:
        original_data = json.load(f)
    
    original_count = len(original_data)
    print(f"✓ Original dataset: {original_count} questions")
    print()
    
    # Get original question IDs
    original_ids = set()
    for sample in original_data:
        if '_id' in sample:
            original_ids.add(sample['_id'])
        elif 'id' in sample:
            original_ids.add(sample['id'])
    
    print(f"✓ Original unique IDs: {len(original_ids)}")
    print()
    
    # Load all split files
    print("Loading split files...")
    all_split_data = []
    all_split_ids = []
    split_files_found = []
    
    for i in range(1, num_splits + 1):
        split_path = os.path.join(splits_dir, f'hotpot_dev_distractor_v1_part{i}.json')
        
        if not os.path.exists(split_path):
            print(f"✗ Warning: Split file {i} not found: {split_path}")
            continue
        
        with open(split_path) as f:
            split_data = json.load(f)
        
        split_count = len(split_data)
        all_split_data.extend(split_data)
        split_files_found.append(i)
        
        # Get IDs from this split
        split_ids = []
        for sample in split_data:
            if '_id' in sample:
                split_ids.append(sample['_id'])
            elif 'id' in sample:
                split_ids.append(sample['id'])
        
        all_split_ids.extend(split_ids)
        
        print(f"  Part {i}: {split_count} questions → {split_path}")
    
    print()
    
    # Verify counts
    total_split_count = len(all_split_data)
    print(f"Total questions in splits: {total_split_count}")
    print()
    
    # Check 1: Total count matches
    print("=" * 80)
    print("Verification Checks")
    print("=" * 80)
    
    if total_split_count == original_count:
        print(f"✓ PASS: Total count matches ({total_split_count} == {original_count})")
    else:
        print(f"✗ FAIL: Total count mismatch ({total_split_count} != {original_count})")
        print(f"  Difference: {abs(total_split_count - original_count)} questions")
        return False
    
    # Check 2: All original IDs are in splits
    split_ids_set = set(all_split_ids)
    missing_ids = original_ids - split_ids_set
    
    if len(missing_ids) == 0:
        print(f"✓ PASS: All original question IDs found in splits")
    else:
        print(f"✗ FAIL: {len(missing_ids)} question IDs missing from splits")
        print(f"  Missing IDs (first 10): {list(missing_ids)[:10]}")
        return False
    
    # Check 3: No duplicate questions across splits
    id_counts = Counter(all_split_ids)
    duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
    
    if len(duplicates) == 0:
        print(f"✓ PASS: No duplicate questions across splits")
    else:
        print(f"✗ FAIL: {len(duplicates)} questions appear in multiple splits")
        print(f"  Duplicate IDs (first 10): {list(duplicates.keys())[:10]}")
        return False
    
    # Check 4: No extra questions in splits
    extra_ids = split_ids_set - original_ids
    
    if len(extra_ids) == 0:
        print(f"✓ PASS: No extra questions in splits (all questions from original)")
    else:
        print(f"✗ WARNING: {len(extra_ids)} question IDs in splits not in original")
        print(f"  Extra IDs (first 10): {list(extra_ids)[:10]}")
        # This is a warning, not a failure
    
    # Check 5: Verify question content matches (sample check)
    print()
    print("=" * 80)
    print("Content Verification (Sample Check)")
    print("=" * 80)
    
    # Create a mapping of original data by ID
    original_by_id = {}
    for sample in original_data:
        if '_id' in sample:
            original_by_id[sample['_id']] = sample
        elif 'id' in sample:
            original_by_id[sample['id']] = sample
    
    # Check first 10 questions match
    matches = 0
    checked = 0
    for sample in all_split_data[:10]:
        sample_id = sample.get('_id') or sample.get('id')
        if sample_id in original_by_id:
            original_sample = original_by_id[sample_id]
            # Compare key fields
            if (sample.get('question') == original_sample.get('question') and
                sample.get('answer') == original_sample.get('answer')):
                matches += 1
            checked += 1
    
    if checked > 0:
        print(f"✓ Sample check: {matches}/{checked} questions match content")
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Original file: {original_path}")
    print(f"  Questions: {original_count}")
    print()
    print(f"Split files found: {len(split_files_found)}/{num_splits}")
    for i in split_files_found:
        split_path = os.path.join(splits_dir, f'hotpot_dev_distractor_v1_part{i}.json')
        with open(split_path) as f:
            split_data = json.load(f)
        print(f"  Part {i}: {len(split_data)} questions")
    print()
    print(f"Total in splits: {total_split_count}")
    print()
    
    if (total_split_count == original_count and 
        len(missing_ids) == 0 and 
        len(duplicates) == 0):
        print("✅ ALL CHECKS PASSED!")
        print("Split files are valid and match the original dataset.")
        return True
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please review the errors above.")
        return False

if __name__ == '__main__':
    # Default paths
    original_path = '../../data/hotpotqa/hotpot_dev_distractor_v1.json'
    splits_dir = '../../data/hotpotqa/splits'
    num_splits = 5
    
    success = verify_splits(original_path, splits_dir, num_splits)
    
    if success:
        print()
        print("=" * 80)
        print("✅ Verification Complete - Splits are valid!")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print("❌ Verification Failed - Please check the errors above")
        print("=" * 80)
        exit(1)
