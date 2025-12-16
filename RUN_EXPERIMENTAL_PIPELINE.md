# Running Enhanced KG²RAG Pipeline

This guide shows how to run the enhanced pipeline and compare it with the original KG²RAG.

## Quick Start

### 1. Run Enhanced Pipeline (First 100 Questions)

```bash
cd code
python kg_rag_enhanced_100.py
```

This will:

- Test on first 100 questions from HotpotQA
- Use multi-view seed retrieval
- Apply knapsack selection
- Save results to `../output/hotpot/hotpot_dev_distractor_v1_kgrag_enhanced_100.json`

### 2. Compare with Original Results

After running both pipelines:

```bash
cd code
python compare_original_vs_enhanced.py \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_enhanced_100_detailed.json
```

## Command-Line Options

### Basic Usage

```bash
python kg_rag_enhanced_100.py \
    --num_questions 100 \
    --result_path ../output/hotpot/hotpot_dev_distractor_v1_kgrag_enhanced_100.json
```

### Multi-View Retrieval Options

```bash
python kg_rag_enhanced_100.py \
    --fusion_method rrf \          # or "mean"
    --top_n_per_view 10 \          # Passages per sub-question
    --seed_top_k 8                   # Final seed count after MMR
```

### Knapsack Selection Options

```bash
python kg_rag_enhanced_100.py \
    --token_budget 2048 \            # Token budget (default: 2048)
    --use_dp_knapsack                # Use DP (exact) solver, otherwise greedy
```

### Sub-Question Generation Options

```bash
# Use Ollama (default)
python kg_rag_enhanced_100.py \
    --model_name llama3:8b

# Use OpenAI (requires API key)
python kg_rag_enhanced_100.py \
    --use_openai \
    --openai_api_key YOUR_API_KEY
```

## Full Example

```bash
cd code

# Run experimental pipeline
python kg_rag_enhanced_100.py \
    --num_questions 100 \
    --fusion_method rrf \
    --seed_top_k 8 \
    --token_budget 2048 \
    --use_dp_knapsack \
    --result_path ../output/hotpot/hotpot_dev_distractor_v1_kgrag_experimental_100.json

# Compare results
python compare_original_vs_enhanced.py \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json \
    ../output/hotpot/hotpot_dev_distractor_v1_kgrag_experimental_100_detailed.json
```

## Output Files

### Main Results

- `hotpot_dev_distractor_v1_kgrag_experimental_100.json`
  - Standard format: `{"answer": {...}, "sp": {...}}`

### Detailed Results

- `hotpot_dev_distractor_v1_kgrag_experimental_100_detailed.json`
  - Includes question info, predictions, supporting facts
  - Includes metadata: sub-questions, seed counts, selection stats

## What's Different from Original

| Stage              | Original KG²RAG        | Experimental KG²RAG           |
| ------------------ | ---------------------- | ----------------------------- |
| **Seed Retrieval** | Single query           | Multi-view (per sub-question) |
| **Fusion**         | N/A                    | RRF or mean-cosine            |
| **Reranking**      | FlagReranker           | Cross-encoder                 |
| **Diversity**      | Graph-based            | MMR filter                    |
| **Selection**      | Graph ordering + top-M | 0-1 knapsack optimization     |
| **Value**          | Graph heuristics       | CE × (1 + coverage)           |
| **Constraint**     | Top-M count            | Token budget                  |

## Expected Performance

The experimental pipeline may show:

- **Better hop coverage** (multi-view ensures both hops)
- **Better token efficiency** (knapsack optimizes value/token)
- **Similar or better accuracy** (depends on question types)

## Troubleshooting

### Import Errors

```bash
# Install required packages
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### Memory Issues

- Reduce `--token_budget` (e.g., 1024 instead of 2048)
- Use `--use_dp_knapsack` only for small candidate pools
- Reduce `--top_n_per_view` (e.g., 5 instead of 10)

### Slow Performance

- Use greedy knapsack (remove `--use_dp_knapsack`)
- Reduce number of sub-questions generated
- Use smaller cross-encoder model

## Next Steps

1. Run enhanced pipeline on 100 questions
2. Compare with original results
3. Analyze improvements/degradations
4. Tune hyperparameters if needed
5. Run on full dataset if results are promising
