# Experimental Extensions

This directory contains **our experimental extensions** to the original KG²RAG framework.

## Overview

We propose two novel improvements:

1. **Multi-View Seed Retrieval**: Retrieve passages for each sub-question separately and fuse using RRF, cross-encoder reranking, and MMR
2. **Token-Budgeted Knapsack Selection**: Replace heuristic top-M selection with 0-1 knapsack optimization

## Main Pipeline Files

- `kg_rag_enhanced.py`: **Experimental pipeline** with multi-view retrieval + knapsack selection
- `kg_rag_baseline.py`: **Baseline pipeline** (original KG²RAG modified for fair comparison)

## Core Components

- `multi_view_retrieval.py`: Multi-view seed retrieval implementation
- `knapsack_selection.py`: Token-budgeted knapsack selection
- `subquestion_generation.py`: Sub-question generation for HotpotQA

## Experiment Management

- `run_experiments.py`: Unified experiment runner (runs both pipelines on same questions)
- `run_experiments_batch.py`: Batch processing for distributed evaluation
- `compile_batches.py`: Compile results from multiple batches
- `enhanced_evaluation.py`: Comprehensive evaluation with confidence intervals
- `generate_report.py`: Generate comparison reports
- `compare_results.py`: Quick comparison script
- `analyze_data_counts.py`: Dataset and KG statistics

## Quick Start

```bash
# Run experiment on 100 questions
python run_experiments.py --num_questions 100 --first

# Generate comparison report
python generate_report.py --output_dir ../../output/hotpot
```

## Documentation

- `BATCH_EVALUATION_GUIDE.md`: Guide for distributed batch evaluation
- `PROJECT_IMPLEMENTATION_GUIDE.md`: Implementation details

For detailed setup and usage, see the main `README_DETAILED.md` in the project root.
