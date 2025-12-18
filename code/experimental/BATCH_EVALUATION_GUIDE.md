# Batch Evaluation Guide

This guide explains how to run experiments in batches (for multiple teammates) and compile results together.

## Overview

When testing on the full dataset (~7,405 questions), you can split the work across multiple teammates:

1. **Each teammate runs batches** using `run_experiments.py`
2. **Compile batches** using `compile_batches.py`
3. **Evaluate compiled results** using `enhanced_evaluation.py`
4. **Generate final report** using `generate_report.py`

## Step-by-Step Process

### Step 1: Run Batches (Each Teammate)

### Option A: Batch-Based (Recommended - Zero Overlap)

Each teammate runs experiments on their assigned batch with **guaranteed zero overlap**:

```bash
# Teammate 1: Questions 0-2500
python run_experiments_batch.py --num_questions 2500 --batch_id 0 --total_batches 3

# Teammate 2: Questions 2500-5000
python run_experiments_batch.py --num_questions 2500 --batch_id 1 --total_batches 3

# Teammate 3: Questions 5000-7500
python run_experiments_batch.py --num_questions 2500 --batch_id 2 --total_batches 3
```

**Benefits:**

- ✅ **Zero overlap** - Each batch gets unique questions
- ✅ **Deterministic** - Same batch_id always gets same questions
- ✅ **Complete coverage** - All batches together cover full dataset

### Option B: Random with Different Seeds (Some Overlap)

If you prefer random sampling (accepts ~10-15% overlap):

```bash
# Teammate 1: First 2500 questions
python run_experiments.py --num_questions 2500 --first

# Teammate 2: Next 2500 questions (requires modifying code to skip first 2500)
# ... or use random sampling with different seeds
```

### Step 2: Compile Batches

After all teammates finish, compile their results:

```bash
# Compile original pipeline batches (batch-based)
python compile_batches.py \
  --batch_dirs exp_2500_batch0of3 exp_2500_batch1of3 exp_2500_batch2of3 \
  --pipeline_type original \
  --output_dir compiled_results \
  --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json

# Compile experimental pipeline batches
python compile_batches.py \
  --batch_dirs exp_2500_batch0of3 exp_2500_batch1of3 exp_2500_batch2of3 \
  --pipeline_type experimental \
  --output_dir compiled_results \
  --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json
```

This creates:

- `compiled_results/original_compiled.json` - Merged predictions
- `compiled_results/original_compiled_detailed.json` - Detailed results with metadata
- `compiled_results/original_batch_stats.json` - Statistics across batches
- Same for `experimental_*`

### Step 3: Enhanced Evaluation

Evaluate the compiled results with comprehensive metrics:

```bash
# Evaluate original pipeline
python enhanced_evaluation.py \
  --prediction compiled_results/original_compiled.json \
  --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json \
  --metadata compiled_results/original_compiled_detailed.json \
  --output compiled_results/original_evaluation.json

# Evaluate experimental pipeline
python enhanced_evaluation.py \
  --prediction compiled_results/experimental_compiled.json \
  --gold ../data/hotpotqa/hotpot_dev_distractor_v1.json \
  --metadata compiled_results/experimental_compiled_detailed.json \
  --output compiled_results/experimental_evaluation.json
```

This generates:

- **Answer quality**: EM, F1, precision, recall
- **Supporting facts quality**: EM, F1, precision, recall
- **Joint metrics**: Answer + supporting facts combined
- **Token efficiency**: Accuracy per token used
- **Failure analysis**: Where each pipeline fails

### Step 4: Generate Final Report

Create a comparison report:

```bash
python generate_report.py \
  --output_dir compiled_results \
  --report_file final_comparison_report.md
```

## Batch Statistics

The `compile_batches.py` script calculates statistics across batches:

- **Mean accuracy**: Average accuracy across all batches
- **Std accuracy**: Standard deviation (shows consistency)
- **Min/Max accuracy**: Range of performance

This helps identify if one batch had issues or if performance is consistent.

## Important Notes

1. **Question IDs**: Make sure batches don't have overlapping question IDs (unless intentional)
2. **Metadata**: Token counts and other metadata are preserved during compilation
3. **Gold File**: Needed for batch statistics calculation
4. **File Structure**: Each batch should be in its own folder (created by `run_experiments.py`)

## Example Workflow

```bash
# === Setup (once) ===
# All teammates have the same codebase and KGs

# === Teammate 1 ===
cd /path/to/KG2RAG-main
python run_experiments_batch.py --num_questions 2500 --batch_id 0 --total_batches 3
# Output: output/hotpot/exp_2500_batch0of3/
# Questions: 0-2500

# === Teammate 2 ===
cd /path/to/KG2RAG-main
python run_experiments_batch.py --num_questions 2500 --batch_id 1 --total_batches 3
# Output: output/hotpot/exp_2500_batch1of3/
# Questions: 2500-5000

# === Teammate 3 ===
cd /path/to/KG2RAG-main
python run_experiments_batch.py --num_questions 2500 --batch_id 2 --total_batches 3
# Output: output/hotpot/exp_2500_batch2of3/
# Questions: 5000-7500

# === Compilation (one person) ===
cd /path/to/KG2RAG-main
python compile_batches.py \
  --batch_dirs output/hotpot/exp_2500_batch0of3 \
              output/hotpot/exp_2500_batch1of3 \
              output/hotpot/exp_2500_batch2of3 \
  --pipeline_type original \
  --output_dir compiled_results \
  --gold data/hotpotqa/hotpot_dev_distractor_v1.json

# === Evaluation ===
python enhanced_evaluation.py \
  --prediction compiled_results/original_compiled.json \
  --gold data/hotpotqa/hotpot_dev_distractor_v1.json \
  --metadata compiled_results/original_compiled_detailed.json \
  --output compiled_results/original_evaluation.json
```

## Troubleshooting

**Issue**: "Batch directory not found"

- **Solution**: Use absolute paths or ensure you're in the correct directory

**Issue**: "Duplicate question IDs"

- **Solution**: Check that batches use different seeds or question ranges

**Issue**: "Missing metadata"

- **Solution**: Ensure `*_detailed.json` files exist in batch folders

**Issue**: "Token counts are 0"

- **Solution**: Check that pipelines were run with the updated code that tracks tokens
