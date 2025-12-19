# Knowledge Graph-Guided Retrieval Augmented Generation: Multi-View Retrieval and Token-Budgeted Selection

This repository contains **experimental extensions** to the original KG²RAG framework from:

**Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, Wei Hu. Knowledge Graph-Guided Retrieval Augmented Generation, NAACL 2025.**

**Experimental Extensions by (Teammates):**

- **Sharvi Endait** - sendait@umass.edu
- **Aditi Kajale** - aditiyogeshk@umass.edu
- **Deva Anand** - devaanand@umass.edu

This repository extends the original KG²RAG framework with two novel improvements for multi-hop question answering:

1. **Multi-View Seed Retrieval**: Retrieve passages for each sub-question separately and fuse using Reciprocal Rank Fusion (RRF), cross-encoder reranking, and Maximal Marginal Relevance (MMR) for diversity.

2. **Token-Budgeted Knapsack Selection**: Replace heuristic top-M selection with a 0-1 knapsack optimization that maximizes value (relevance × coverage) under token budget constraints.

![image](framework.jpg)

## Repository Structure

```
KG2RAG-main/
├── code/
│   ├── original/              # Original KG²RAG implementation
│   │   ├── kg_rag_distractor.py
│   │   ├── kg_rag_full.py
│   │   └── README.md
│   ├── experimental/         # Our experimental extensions
│   │   ├── kg_rag_enhanced.py      # Experimental pipeline
│   │   ├── kg_rag_baseline.py       # Baseline (for comparison)
│   │   ├── multi_view_retrieval.py  # Multi-view retrieval
│   │   ├── knapsack_selection.py    # Knapsack selection
│   │   ├── subquestion_generation.py
│   │   ├── run_experiments.py       # Experiment runner
│   │   └── README.md
│   ├── util/                 # Shared utilities (KG expansion, answer generation)
│   └── preprocess/           # Data preprocessing (KG extraction)
├── data/                     # Datasets and extracted KGs
├── output/                   # Experiment results
├── README.md                 # This file
├── CODE_DOCUMENTATION.md     # Code structure and documentation
├── SETUP_INSTRUCTIONS.md     # Setup guide
├── RESULTS_AND_ERROR_ANALYSIS.md  # Experimental results
└── requirements.txt          # Python dependencies
```

### Key Differences from Original KG²RAG

| Component            | Original KG²RAG                                  | Experimental Approach                                                                                             |
| -------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Seed Retrieval**   | Single-view semantic retrieval                   | Multi-view retrieval (one view per sub-question) with RRF/mean fusion, cross-encoder reranking, and MMR diversity |
| **Sub-Questions**    | Not explicitly used                              | Generated for HotpotQA (or extracted for MuSiQue) to enable multi-view retrieval                                  |
| **Selection Method** | Graph-based ordering (MST/DFS) + heuristic top-M | Token-budgeted 0-1 knapsack optimization maximizing (relevance × coverage)                                        |
| **Candidate Pool**   | All passages after KG expansion                  | Restricted to multi-view seeds + their 1-hop KG-expanded neighbors                                                |
| **Optimization**     | Graph structure + reranking                      | Explicit token budget constraint with value maximization                                                          |

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup Ollama (if not already installed)
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

### Run Experiments

```bash
cd code/experimental

# Run experiment on 100 questions
python run_experiments.py --num_questions 100 --first

# Generate comparison report
python generate_report.py --output_dir ../../output/hotpot
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running (for LLM inference)
- 16GB+ RAM recommended
- GPU optional but recommended for faster processing

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Step 2: Setup Ollama

```bash
# Install Ollama (if not already installed)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

### Step 3: Prepare Data

The HotpotQA dataset should be in `data/hotpotqa/hotpot_dev_distractor_v1.json`.

For KG extraction (if not already done):

```bash
cd code/preprocess
python hotpot_extraction.py --data_path ../../data/hotpotqa/hotpot_dev_distractor_v1.json --output_dir ../../data/hotpotqa/kgs/extract_subkgs
```

## Usage

### Quick Start: Run Experiments

```bash
cd code/experimental

# Run experiment on 100 questions
python run_experiments.py --num_questions 100 --first

# Run experiment on random 100 questions (with seed for reproducibility)
python run_experiments.py --num_questions 100 --random --seed 42

# Generate comparison report
python generate_report.py --output_dir ../../output/hotpot
```

### Running Original vs Experimental Pipelines

The `run_experiments.py` script runs both pipelines on the same questions for fair comparison:

```bash
# Test on first 250 questions
python run_experiments.py --num_questions 250 --first

# Test on random 500 questions
python run_experiments.py --num_questions 500 --random --seed 42
```

Results are saved in `output/hotpot/exp_<N>_<type>/` with:

- `original.json` and `original_detailed.json` - Original pipeline results
- `experimental.json` and `experimental_detailed.json` - Experimental pipeline results
- `metadata.json` - Token counts and other metadata

### Batch Processing (For Distributed Evaluation)

For large-scale evaluation across multiple machines:

```bash
# Teammate 1: Questions 0-2500
python run_experiments_batch.py --num_questions 2500 --batch_id 0 --total_batches 3

# Teammate 2: Questions 2500-5000
python run_experiments_batch.py --num_questions 2500 --batch_id 1 --total_batches 3

# Teammate 3: Questions 5000-7500
python run_experiments_batch.py --num_questions 2500 --batch_id 2 --total_batches 3
```

Then compile results:

```bash
python compile_batches.py \
  --batch_dirs exp_2500_batch0of3 exp_2500_batch1of3 exp_2500_batch2of3 \
  --pipeline_type original --output_dir compiled_full \
  --gold ../../data/hotpotqa/hotpot_dev_distractor_v1.json --base_dir ../../output/hotpot
```

## Documentation

For detailed information, see:

- **[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)** - Code structure and documentation
- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Setup guide for teammates
- **[RESULTS_AND_ERROR_ANALYSIS.md](RESULTS_AND_ERROR_ANALYSIS.md)** - Experimental results and analysis
- **[code/original/README.md](code/original/README.md)** - Original KG²RAG code documentation
- **[code/experimental/README.md](code/experimental/README.md)** - Experimental extensions documentation
- **[code/experimental/EXPERIMENT_GUIDE.md](code/experimental/EXPERIMENT_GUIDE.md)** - Experiment guide

## Contact

For questions about the experimental extensions, please contact:

- **Sharvi Endait** - sendait@umass.edu
- **Aditi Kajale** - aditiyogeshk@umass.edu
- **Deva Anand** - devaanand@umass.edu

For questions about the original KG²RAG paper -> please contact: xrzhu.nju@gmail.com
