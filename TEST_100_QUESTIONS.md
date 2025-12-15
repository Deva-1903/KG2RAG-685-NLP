# Test KG²RAG on 100 Questions

This guide shows how to test the original pipeline on 100 questions only.

## Quick Start

```bash
cd code
python kg_rag_distractor_100.py
```

This will:

- Test on first 100 questions
- Use original pipeline (no modifications)
- Save results with question tracking
- Show which questions were tested

## Customize Number of Questions

```bash
# Test 50 questions
python kg_rag_distractor_100.py --num_questions 50

# Test 200 questions
python kg_rag_distractor_100.py --num_questions 200
```

## Output Files

### 1. Main Results (`hotpot_dev_distractor_v1_kgrag_100.json`)

Standard format (same as original):

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

### 2. Detailed Results (`hotpot_dev_distractor_v1_kgrag_100_detailed.json`)

Includes question info and ground truth:

```json
{
  "summary": {
    "total_tested": 100,
    "questions_tested": [
      {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood...",
        "ground_truth": "yes"
      },
      ...
    ]
  },
  "results": [
    {
      "question_id": "5a8b57f25542995d1e6f1371",
      "question": "Were Scott Derrickson and Ed Wood...",
      "ground_truth": "yes",
      "prediction": "Yes.",
      "supporting_facts": [["Scott Derrickson", 0], ...]
    },
    ...
  ]
}
```

## What's Different from Original

**Only change**: Limits dataset to first N questions (default 100)

**Everything else**: Identical to original `kg_rag_distractor.py`

- Same pipeline
- Same post-processors
- Same reranker
- Same answer generation

## Time Estimate

- **100 questions**: ~20-40 minutes (CPU) or ~10-20 minutes (GPU)
- Much faster than full 7,405 questions!

## View Results

```bash
# View detailed results
cat ../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json | python -m json.tool | less

# Or use Python
python -c "
import json
with open('../output/hotpot/hotpot_dev_distractor_v1_kgrag_100_detailed.json') as f:
    data = json.load(f)
    print(f'Tested {data[\"summary\"][\"total_tested\"]} questions')
    print('\\nFirst 3 results:')
    for r in data['results'][:3]:
        print(f\"\\nQ: {r['question']}\")
        print(f\"Truth: {r['ground_truth']}\")
        print(f\"Pred: {r['prediction']}\")
"
```

## Compare with Original

The original pipeline (`kg_rag_distractor.py`) processes all questions.
This test version (`kg_rag_distractor_100.py`) processes only first 100.

Both use the **exact same pipeline** - just different dataset size!
