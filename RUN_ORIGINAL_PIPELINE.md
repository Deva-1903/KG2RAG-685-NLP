# Run Original KG²RAG Pipeline

This guide shows how to run the **original, unmodified** KG²RAG pipeline with your current partial KG files.

## Prerequisites

1. **Ollama running**:

   ```bash
   ollama serve
   ```

2. **Models downloaded**:

   ```bash
   ollama pull llama3:8b
   ollama pull mxbai-embed-large
   ```

3. **KG files available**:
   - Location: `data/hotpotqa/kgs/extract_subkgs/`
   - You have: ~28,492 KG files (partial coverage)

## Run Original Pipeline

### Basic Command (Uses Defaults)

```bash
cd code
python kg_rag_distractor.py
```

This uses:

- Dataset: `hotpotqa`
- Data: `../data/hotpotqa/hotpot_dev_distractor_v1.json`
- KG Directory: `../data/hotpotqa/kgs/extract_subkgs`
- Output: `../output/hotpot/hotpot_dev_distractor_v1_kgrag.json`

### Custom Arguments

```bash
cd code
python kg_rag_distractor.py \
  --dataset hotpotqa \
  --data_path ../data/hotpotqa/hotpot_dev_distractor_v1.json \
  --kg_dir ../data/hotpotqa/kgs/extract_subkgs \
  --result_path ../output/hotpot/my_results.json \
  --top_k 10
```

## What Happens

1. **Loads dataset**: All 7,405 questions from HotpotQA
2. **Loads KGs**: Only entities that have KG files (your ~28,492 files)
3. **Processes each question**:
   - Questions with KG coverage → Uses KG²RAG (full pipeline)
   - Questions without KG coverage → Falls back to semantic-only retrieval
4. **Saves results**: JSON file with predictions

## Output Format

```json
{
  "answer": {
    "question_id": "predicted answer",
    ...
  },
  "sp": {
    "question_id": [["Entity1", 0], ["Entity2", 1]],
    ...
  }
}
```

## Notes

- **Partial KG coverage is fine**: The system handles missing KGs gracefully
- **No modifications needed**: Original code works with partial data
- **Results will be mixed**: Some questions use KG²RAG, others use semantic-only
- **Time**: ~2-4 hours (GPU) or 6-10 hours (CPU) for all 7,405 questions

## Troubleshooting

### FlagEmbedding Import Error

If you get FlagEmbedding errors, the system will still work but without reranking. To fix:

```bash
pip install transformers==4.35.0 FlagEmbedding==1.2.11
```

### Check KG Coverage

```bash
cd code
python -c "
import os
kg_dir = '../data/hotpotqa/kgs/extract_subkgs'
kg_files = [f for f in os.listdir(kg_dir) if f.endswith('.json')]
print(f'KG files available: {len(kg_files)}')
"
```

## That's It!

The original pipeline is ready to run. Just execute:

```bash
cd code
python kg_rag_distractor.py
```

The system will automatically use whatever KG files you have available.

