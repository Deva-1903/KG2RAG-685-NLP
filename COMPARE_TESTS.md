# Compare First 100 vs Random 100 Questions

This guide shows how to test on random 100 questions and compare results with the first 100 questions.

## Step 1: Run Test on Random 100 Questions

```bash
cd code
python kg_rag_distractor_100.py \
    --random_sample \
    --seed 42 \
    --result_path ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random.json \
    --num_questions 100
```

**What this does:**

- `--random_sample`: Randomly sample questions instead of taking first 100
- `--seed 42`: Use seed 42 for reproducibility (you can change this)
- `--result_path`: Save to a different file so we don't overwrite the first 100 results
- `--num_questions 100`: Test 100 questions

**Output files:**

- `../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random.json`
- `../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random_detailed.json`

## Step 2: Compare Results

After both tests are complete, compare them:

```bash
cd code
python compare_results.py \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random_detailed.json \
    "First 100" \
    "Random 100"
```

**What the comparison shows:**

- Individual metrics for both test sets
- Accuracy differences
- Question type breakdown (Yes/No vs Factoid)
- Supporting facts statistics
- Question overlap (how many questions appear in both sets)

## Example Output

```
================================================================================
KG²RAG Results Comparison
================================================================================

First 100 Results:
--------------------------------------------------------------------------------
  Total Questions: 100
  Exact Matches: 45 (45.0%)
  Partial Matches: 20 (20.0%)
  Overall Accuracy: 65.0%
  Yes/No Questions: 25 (Accuracy: 80.0%)
  Factoid Questions: 75 (Accuracy: 60.0%)
  Avg Supporting Facts: 7.2

Random 100 Results:
--------------------------------------------------------------------------------
  Total Questions: 100
  Exact Matches: 48 (48.0%)
  Partial Matches: 18 (18.0%)
  Overall Accuracy: 66.0%
  Yes/No Questions: 23 (Accuracy: 82.6%)
  Factoid Questions: 77 (Accuracy: 61.0%)
  Avg Supporting Facts: 7.5

Comparison:
--------------------------------------------------------------------------------
  Overall Accuracy Difference: +1.0% (Random 100 - First 100)
  Exact Match Difference: +3.0% (Random 100 - First 100)
  Yes/No Accuracy Difference: +2.6% (Random 100 - First 100)
  Factoid Accuracy Difference: +1.0% (Random 100 - First 100)
  Question Overlap: 0 questions appear in both sets

Summary:
--------------------------------------------------------------------------------
  ≈ Both sets perform similarly (difference: 1.0%)
================================================================================
```

## Why Compare?

Comparing first 100 vs random 100 helps you understand:

1. **Consistency**: Does the model perform similarly across different question sets?
2. **Bias**: Are the first 100 questions easier/harder than average?
3. **Generalization**: How well does the model generalize to different questions?

## Notes

- The random sample uses seed 42 by default (change with `--seed`)
- Both test sets should have 0 overlap (different questions)
- The comparison script shows detailed metrics for both sets
- You can use different seeds to test multiple random samples

## Quick Commands

**Test random 100:**

```bash
cd code
python kg_rag_distractor_100.py --random_sample --result_path ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random.json
```

**Compare results:**

```bash
cd code
python compare_results.py \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_random_detailed.json
```
