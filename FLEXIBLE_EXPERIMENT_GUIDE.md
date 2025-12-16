# Flexible Experiment System Guide

## Overview

The flexible experiment system allows you to:

- Run experiments with **any number of questions** (100, 200, 300, 400, 500, etc.)
- Choose between **random sampling** or **first N** questions
- **Automatically ensure same random seed** for both pipelines when using random sampling
- **Organize results in folders** for easy management
- **Generate comprehensive reports** from all experiments

---

## Quick Start

### Run a Single Experiment

```bash
cd code

# Random 100 questions
python run_experiments.py --num_questions 100 --random

# First 200 questions
python run_experiments.py --num_questions 200 --first

# Random 500 questions with specific seed
python run_experiments.py --num_questions 500 --random --seed 123
```

### Run Multiple Experiments

```bash
# Run multiple experiments at once
python run_experiments.py --num_questions 100 200 300 --random

# Mix different configurations
python run_experiments.py --num_questions 100 200 --first
python run_experiments.py --num_questions 300 400 500 --random --seed 42
```

### Generate Report

```bash
# Generate report from all experiments
python generate_report.py --output_dir ../output/hotpot

# Generate report for a specific experiment folder
python generate_report.py --output_dir ../output/hotpot/exp_100_first

# Save report to file
python generate_report.py --output_dir ../output/hotpot --report_file ../experiment_report.md
```

---

## Folder Structure

Results are organized in **experiment folders** with simple filenames:

### Random Sampling

```
output/hotpot/
└── exp_100_random_seed42/
    ├── original.json
    ├── original_detailed.json
    ├── experimental.json
    ├── experimental_detailed.json
    └── metadata.json
```

### First N Sampling

```
output/hotpot/
└── exp_200_first/
    ├── original.json
    ├── original_detailed.json
    ├── experimental.json
    ├── experimental_detailed.json
    └── metadata.json
```

**Key Points:**

- Each experiment has its own folder
- Simple, consistent filenames inside folders
- Folder name encodes configuration (num_questions, sampling, seed)
- Easy to find, compare, and delete experiments

---

## Detailed Usage

### `run_experiments.py`

**Purpose:** Run both original and experimental pipelines with the same configuration.

**Arguments:**

- `--num_questions N [N ...]`: Number of questions (can specify multiple)
- `--random`: Use random sampling
- `--first`: Use first N questions
- `--seed SEED`: Random seed (default: 42 if using random)
- `--output_dir DIR`: Output directory (default: ../output/hotpot)

**Examples:**

```bash
# Single experiment: random 100
python run_experiments.py --num_questions 100 --random

# Single experiment: first 200
python run_experiments.py --num_questions 200 --first

# Multiple experiments: random 100, 200, 300
python run_experiments.py --num_questions 100 200 300 --random

# Custom seed
python run_experiments.py --num_questions 500 --random --seed 999

# Custom output directory
python run_experiments.py --num_questions 100 --random --output_dir ../output/custom
```

**What it does:**

1. Validates configuration
2. Creates experiment folder (e.g., `exp_100_random_seed42/`)
3. Runs original pipeline with specified config
4. Runs experimental pipeline with **same config** (same seed for random)
5. Saves results with simple filenames in the folder
6. Saves experiment metadata
7. Prints summary and comparison command

---

### `generate_report.py`

**Purpose:** Analyze all experiment results and generate comprehensive report.

**Arguments:**

- `--output_dir DIR`: Directory containing experiment results (default: ../output/hotpot)
- `--report_file FILE`: Save report to file (optional)

**Examples:**

```bash
# Generate and print report
python generate_report.py --output_dir ../output/hotpot

# Generate and save report
python generate_report.py --output_dir ../output/hotpot --report_file ../experiment_report.md
```

**Report includes:**

- Summary table comparing all experiments
- Detailed metrics for each experiment
- Improvement analysis
- Overall statistics (average, max, min improvements)

**How it works:**

- Automatically scans for `exp_*` folders
- Groups original and experimental results by configuration
- Can process a specific folder or all folders in a directory

---

## Example Workflow

### Scenario: Run Multiple Experiments and Generate Report

```bash
cd code

# Step 1: Run experiments
python run_experiments.py --num_questions 100 200 300 --random --seed 42
python run_experiments.py --num_questions 100 200 300 --first

# Step 2: Generate comprehensive report
python generate_report.py --output_dir ../output/hotpot --report_file ../experiment_report.md

# Step 3: View report
cat ../experiment_report.md
```

### Scenario: Compare Specific Experiments

```bash
# Run specific experiments
python run_experiments.py --num_questions 500 --random --seed 123

# Compare results
python generate_report.py --output_dir ../output/hotpot
```

---

## Ensuring Same Random Questions

**Key Feature:** When using `--random`, both pipelines automatically use the **same seed**.

**How it works:**

1. If you specify `--seed 123`, both pipelines use seed 123
2. If you don't specify seed, both use default seed 42
3. This ensures **identical question selection** for fair comparison

**Example:**

```bash
# Both pipelines will test on the SAME 100 random questions
python run_experiments.py --num_questions 100 --random --seed 123
```

**Verification:**
Check the question IDs in the detailed JSON files - they should match exactly:

```bash
python -c "
import json
orig = json.load(open('../output/hotpot/exp_100_random_seed123/original_detailed.json'))
exp = json.load(open('../output/hotpot/exp_100_random_seed123/experimental_detailed.json'))
orig_ids = [q['id'] for q in orig['summary']['questions_tested']]
exp_ids = [q['id'] for q in exp['summary']['questions_tested']]
print('Same questions:', orig_ids == exp_ids)
print('First 5 IDs match:', orig_ids[:5] == exp_ids[:5])
"
```

---

## Output Files Structure

```
output/hotpot/
├── exp_100_random_seed42/
│   ├── original.json
│   ├── original_detailed.json
│   ├── experimental.json
│   ├── experimental_detailed.json
│   └── metadata.json
├── exp_200_first/
│   ├── original.json
│   ├── original_detailed.json
│   ├── experimental.json
│   ├── experimental_detailed.json
│   └── metadata.json
└── exp_500_random_seed123/
    ├── original.json
    ├── original_detailed.json
    ├── experimental.json
    ├── experimental_detailed.json
    └── metadata.json
```

**File Contents:**

- `original.json` / `experimental.json`: Main results (answers and supporting facts)
- `original_detailed.json` / `experimental_detailed.json`: Detailed results with questions and metadata
- `metadata.json`: Experiment configuration and file paths

**Benefits:**

- ✅ Easy to find experiments (just look for folder name)
- ✅ Easy to delete experiments (remove folder)
- ✅ Clean organization (no long filenames)
- ✅ Simple to compare (same filenames in each folder)

---

## Advanced Usage

### Running Experiments in Parallel

Since experiments are independent, you can run multiple in parallel:

```bash
# Terminal 1
python run_experiments.py --num_questions 100 --random --seed 42

# Terminal 2
python run_experiments.py --num_questions 200 --random --seed 42

# Terminal 3
python run_experiments.py --num_questions 300 --random --seed 42
```

### Custom Configurations

You can modify `run_experiments.py` to add custom configurations:

- Different models
- Different hyperparameters
- Different datasets

---

## Troubleshooting

### Different Question IDs

**Problem:** Question IDs don't match between original and experimental.

**Solution:**

- Ensure both used `--random` flag
- Check that both used the same `--seed` value
- Verify both loaded from the same data file

### Missing Results

**Problem:** Some experiment results are missing.

**Solution:**

- Check that pipelines completed successfully
- Verify output directory exists
- Check for error messages in terminal

### Report Generation Fails

**Problem:** `generate_report.py` can't find experiments.

**Solution:**

- Verify `--output_dir` points to correct directory (parent directory or specific experiment folder)
- Check that experiment folders start with `exp_`
- Ensure `original_detailed.json` and `experimental_detailed.json` exist in folders
- Try specifying a specific experiment folder: `--output_dir ../output/hotpot/exp_100_first`

---

## Best Practices

1. **Use consistent seeds** for reproducibility
2. **Run multiple experiments** to verify consistency
3. **Generate reports regularly** to track progress
4. **Save reports** for documentation
5. **Use descriptive seeds** (e.g., seed 42 for baseline, seed 123 for validation)

---

## Summary

✅ **Flexible:** Any number of questions (100, 200, 300, etc.)  
✅ **Same Random Seed:** Both pipelines use identical seed automatically  
✅ **Folder Organization:** Each experiment in its own folder  
✅ **Simple Filenames:** Consistent naming inside folders  
✅ **Comprehensive Reports:** Analyze all experiments at once  
✅ **Easy Management:** Find, compare, and delete experiments easily

**Ready to run multiple experiments and generate comprehensive reports!**
