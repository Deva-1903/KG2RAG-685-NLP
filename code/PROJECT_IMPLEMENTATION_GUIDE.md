# Project Implementation Guide

This guide explains how to use the experimental components (multi-view retrieval and knapsack selection) as an alternative approach to the original KG²RAG pipeline.

## Overview

Our project experiments with two alternative approaches to KG²RAG:

1. **Multi-View Seed Retrieval** (`multi_view_retrieval.py`)

   - Retrieve passages for each sub-question separately
   - Fuse using RRF or mean-cosine
   - Rerank with cross-encoder
   - Apply MMR for diversity

2. **Token-Budgeted Knapsack Selection** (`knapsack_selection.py`)

   - Replace heuristic top-M selection with 0-1 knapsack optimization
   - Maximize value (relevance × coverage) under token budget

3. **Sub-Question Generation** (`subquestion_generation.py`)
   - Generate sub-questions for HotpotQA
   - Use existing sub-questions from MuSiQue

## File Structure

```
code/
├── multi_view_retrieval.py      # Experimental #1: Multi-view seed retrieval
├── knapsack_selection.py         # Experimental #2: Token-budgeted selection
├── subquestion_generation.py    # Sub-question generation
├── kg_rag_enhanced_100.py        # Experimental pipeline (tests 100 questions)
└── generate_report.py  # Report generation script
```

## Usage Examples

### 1. Multi-View Retrieval

```python
from multi_view_retrieval import MultiViewRetriever
from llama_index.core import VectorStoreIndex

# Create index (from your corpus)
index = VectorStoreIndex.from_documents(documents)

# Initialize multi-view retriever
retriever = MultiViewRetriever(
    index=index,
    fusion_method="rrf",  # or "mean"
    top_n_per_view=10,
    final_top_k=8
)

# Get sub-questions (from generator or dataset)
sub_questions = ["Who is X?", "What is Y?"]

# Retrieve diverse seeds
seed_passages = retriever.retrieve_seeds(
    main_question="Original question",
    sub_questions=sub_questions
)

# Evaluate seed quality
hop_coverage = retriever.compute_hop_coverage(
    seed_passages,
    gold_supporting_facts=[("Entity1", 0), ("Entity2", 1)]
)

recall_at_k = retriever.compute_recall_at_k(
    seed_passages,
    gold_supporting_facts,
    k=8
)
```

### 2. Knapsack Selection

```python
from knapsack_selection import KnapsackSelector
import spacy

# Load spaCy for entity extraction
nlp = spacy.load("en_core_web_sm")

# Initialize selector
selector = KnapsackSelector(
    token_budget=2048,
    use_dp=True  # Use DP (exact) or greedy (faster)
)

# Extract entities from question
doc = nlp(question)
question_entities = {ent.text.lower() for ent in doc.ents}
subquestion_keywords = set(question.lower().split())

# Candidates after KG expansion
candidates = [
    ("doc1##0", "Passage text 1..."),
    ("doc2##1", "Passage text 2..."),
    # ...
]

# Select evidence
selected_doc_ids, selected_values, total_value = selector.select_evidence(
    question=question,
    candidates=candidates,
    question_entities=question_entities,
    subquestion_keywords=subquestion_keywords
)

# Compare with top-M baseline
comparison = selector.compare_with_top_m(
    question=question,
    candidates=[(doc_id, text, score) for doc_id, text in candidates],
    question_entities=question_entities,
    subquestion_keywords=subquestion_keywords,
    top_m=10
)
```

### 3. Sub-Question Generation

```python
from subquestion_generation import SubQuestionGenerator

# Initialize generator
generator = SubQuestionGenerator(
    use_openai=False,  # Use Ollama instead
    ollama_model="llama3:8b"
)

# For HotpotQA: generate sub-questions
question = "Were Scott Derrickson and Ed Wood of the same nationality?"
sub_questions = generator.get_subquestions(
    question=question,
    dataset="hotpotqa"
)
# Returns: ["Who is Scott Derrickson?", "Who is Ed Wood?", ...]

# For MuSiQue: use existing sub-questions
sample = {
    "decomposition": {
        "questions": ["Sub-question 1", "Sub-question 2"]
    }
}
sub_questions = generator.get_subquestions(
    question=question,
    dataset="musique",
    sample=sample
)
```

## Integration with Original Pipeline

### Original KG²RAG Flow:

1. Single-view retrieval → seeds
2. KG expansion (1-hop)
3. Graph ordering (MST/DFS)
4. Top-M selection
5. Answer generation

### Experimental Flow:

1. **Multi-view retrieval** → diverse seeds (6-8) [experimental]
2. KG expansion (1-hop) [same as original]
3. Candidate scoring (CE × coverage) [experimental]
4. **Knapsack selection** (token-budgeted) [experimental]
5. Answer generation [same as original]

## Evaluation Metrics

### Seed Stage:

- **Recall@k**: Fraction of gold supporting facts in top-k seeds
- **Hop Coverage**: Whether both hops are present
- **Entity Diversity**: Number of unique entities in seeds

### Selection Stage:

- **Token Efficiency**: EM vs tokens/query
- **Evidence Precision/Recall**: vs supporting facts
- **Value Optimization**: Total value vs top-M baseline

## Baselines and Ablations

### Baselines:

- **S0**: Semantic/Hybrid RAG (no KG)
- **S1**: KG²RAG-Lite (single-view, graph ordering, top-M)

### Experimental System:

- **S2**: Multi-view + KG + knapsack (experimental approach)

### Ablations:

- **A1**: Single-view vs multi-view seeds
- **A2**: Top-M vs knapsack at same budget

## Next Steps

1. **Run experimental pipeline** (`kg_rag_enhanced_100.py`) on 100 questions
2. **Compare results** using `generate_report.py`
3. **Analyze differences** between original and experimental approaches
4. **Tune hyperparameters** if needed

## Dependencies

```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

For OpenAI sub-question generation:

```bash
pip install openai
```

## Notes

- Multi-view retrieval ensures both hops are covered before KG expansion
- Knapsack selection optimizes token usage while maximizing value
- Sub-question generation enables multi-view retrieval for HotpotQA
- All components are modular and can be used independently
