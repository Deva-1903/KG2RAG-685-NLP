# Code Documentation Guide

This document provides an overview of code structure, key functions, and implementation details.

## Code Organization

### Main Pipeline Files

#### `kg_rag_enhanced_100.py`

**Purpose**: Main experimental pipeline orchestrator implementing multi-view retrieval + knapsack selection.

**Key Functions**:

- `read_data()`: Load and filter dataset (supports random sampling, batch processing)
- `init_model()`: Initialize Ollama LLM and embeddings
- `read_kg()`: Load extracted knowledge graphs from JSON files
- `build_document_index()`: Create vector index from question's context passages
- `process_sample_enhanced()`: Main processing pipeline for a single question
  - Step 1: Generate sub-questions
  - Step 2: Multi-view seed retrieval
  - Step 3: KG expansion (1-hop)
  - Step 4: Candidate pool construction (cap at 60)
  - Step 5: Knapsack selection
  - Step 6: Answer generation

**Key Parameters**:

- `--num_questions`: Number of questions to test
- `--random_sample`: Use random sampling instead of first N
- `--seed`: Random seed for reproducibility
- `--start_idx`: Starting index for batch processing
- `--token_budget`: Token budget for knapsack (default: 2048)
- `--seed_top_k`: Number of seed passages (default: 8)
- `--fusion_method`: "rrf" or "mean" for multi-view fusion

#### `kg_rag_distractor_100.py`

**Purpose**: Original KG²RAG baseline pipeline (modified for N questions).

**Key Differences from Original**:

- Supports `--num_questions` for testing on subsets
- Supports batch processing with `--start_idx`
- Tracks token usage in metadata
- Same core pipeline as original KG²RAG

### Novel Components

#### `multi_view_retrieval.py`

**Purpose**: Multi-view seed retrieval (Novelty #1).

**Key Classes**:

- `MultiViewRetriever`: Main class for multi-view retrieval

**Key Methods**:

- `retrieve_for_subquestion()`: Retrieve passages for a single sub-question
- `fuse_rankings()`: Fuse multiple rankings using RRF or mean-cosine
- `rerank_with_cross_encoder()`: Rerank candidates using cross-encoder
- `mmr_diversity_filter()`: Apply MMR for diversity

**Algorithm Flow**:

1. For each sub-question: retrieve top-N passages
2. Fuse rankings using RRF: `score += 1.0 / (k + rank)`
3. Rerank fused candidates with cross-encoder
4. Apply MMR: `MMR = λ × relevance - (1-λ) × max_similarity`

#### `knapsack_selection.py`

**Purpose**: Token-budgeted knapsack selection (Novelty #2).

**Key Classes**:

- `KnapsackSelector`: Main class for knapsack selection

**Key Functions**:

- `compute_coverage_bonus()`: Calculate entity coverage = |ent(q) ∩ ent(c)| / |ent(q)|
- `compute_candidate_value()`: Value = CE_score × (1 + coverage)
- `knapsack_01_dp()`: Exact 0-1 knapsack solver (dynamic programming)
- `knapsack_greedy()`: Greedy approximation solver

**Algorithm**:

- **Value**: `v(c) = CE(q, c) × (1 + coverage(c))`
- **Cost**: `w(c) = token_count(c)`
- **Objective**: Maximize Σ v(c) subject to Σ w(c) ≤ budget

#### `subquestion_generation.py`

**Purpose**: Generate sub-questions for multi-hop questions.

**Key Classes**:

- `SubQuestionGenerator`: Generate sub-questions using LLM

**Key Methods**:

- `generate_for_hotpotqa()`: Generate sub-questions for HotpotQA using LLM
- `generate_for_musique()`: Use existing sub-questions from MuSiQue dataset
- `_fallback_generation()`: Heuristic fallback if LLM fails

### Experiment Management

#### `run_experiments.py`

**Purpose**: Unified experiment runner for both pipelines.

**Features**:

- Runs original and experimental pipelines on same questions
- Supports random/sequential sampling
- Automatic output folder naming
- Ensures fair comparison (same questions, same seed)

#### `run_experiments_batch.py`

**Purpose**: Batch processing for distributed evaluation.

**Features**:

- Zero-overlap batch splitting
- Deterministic question assignment
- Supports multiple teammates running in parallel

#### `compile_batches.py`

**Purpose**: Compile results from multiple batches.

**Features**:

- Merges predictions from multiple batches
- Calculates batch-level statistics
- Computes 95% confidence intervals
- Detects and reports duplicates

#### `enhanced_evaluation.py`

**Purpose**: Comprehensive evaluation with advanced metrics.

**Metrics**:

- Answer quality: EM, F1, precision, recall (with 95% CI)
- Supporting facts: EM, F1, precision, recall (with 95% CI)
- Joint metrics: Answer + supporting facts
- Token efficiency: Accuracy per token
- Per-question analysis
- Failure categorization

### Utility Files

#### `util/kg_post_processor.py`

**Purpose**: KG expansion and graph operations (from original KG²RAG).

**Key Classes**:

- `KGRetrievePostProcessor`: 1-hop KG expansion from seed passages
- `GraphFilterPostProcessor`: Graph-based filtering (not used in experimental)

#### `util/kg_response_synthesizer.py`

**Purpose**: Answer generation using LLM (from original KG²RAG).

**Key Functions**:

- `get_response_synthesizer()`: Get LLM response synthesizer

#### `util/hotpot_evaluate_v1.py`

**Purpose**: Evaluation metrics (from original KG²RAG).

**Key Functions**:

- `f1_score()`: Calculate F1 score between prediction and ground truth
- `exact_match_score()`: Calculate exact match score

## Code Comments and Documentation Standards

### Function Docstrings

All functions follow this format:

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised
    """
```

### Inline Comments

- **Complex logic**: Explained with inline comments
- **Algorithm steps**: Numbered or labeled for clarity
- **Non-obvious decisions**: Rationale provided
- **Parameter choices**: Default values explained

### Example: Well-Documented Function

```python
def compute_coverage_bonus(
    passage_text: str,
    question_entities: Set[str],
    subquestion_keywords: Set[str] = None
) -> float:
    """
    Compute coverage bonus: how many question entities appear in passage.

    As per proposal: coverage(c) = |ent(q) ∩ ent(c)| / |ent(q)|

    Args:
        passage_text: Passage text to evaluate
        question_entities: Entities extracted from the question
        subquestion_keywords: (Optional, kept for compatibility) Keywords from sub-questions

    Returns:
        Coverage score [0, 1] where 1.0 means all question entities appear in passage
    """
    if not question_entities:
        return 0.0

    # Extract entities from passage
    passage_entities = extract_entities(passage_text.lower())

    # Calculate entity overlap: intersection / question entities
    entity_overlap = len(question_entities & passage_entities)
    coverage = entity_overlap / len(question_entities)

    return min(coverage, 1.0)  # Cap at 1.0
```

## Key Design Decisions

### 1. Pool Size Capping (MAX_POOL_SIZE = 60)

**Rationale**: Knapsack DP solver has O(n × budget) complexity. Capping at 60 ensures:

- Computational efficiency (solves in <1 second)
- Maintains diversity (60 candidates is sufficient)
- Prioritizes seed passages (higher quality)

### 2. Multi-View Retrieval

**Rationale**: Single-view retrieval may miss one "hop" in multi-hop questions. Multi-view ensures:

- Each sub-question gets its own retrieval pass
- Coverage of all reasoning aspects
- Better recall for complex questions

### 3. Knapsack vs. Top-M

**Rationale**: Top-M selection ignores token costs. Knapsack:

- Optimizes token usage
- Maximizes value (relevance × coverage)
- Better fits within LLM context limits

### 4. Coverage Formula

**Rationale**: Entity-only coverage (not keyword-based) because:

- Entities are more reliable indicators of relevance
- Matches proposal specification
- Simpler and more interpretable

## Testing and Validation

### Unit Testing

Key functions have been tested individually:

- `reciprocal_rank_fusion()`: Tested with known rankings
- `knapsack_01_dp()`: Tested with small examples
- `compute_coverage_bonus()`: Tested with various entity sets

### Integration Testing

End-to-end pipeline tested on:

- 10 questions (quick validation)
- 100 questions (initial evaluation)
- 4,905 questions (batches 0 & 2)

### Validation Checks

- **No duplicate questions**: Batch processing ensures zero overlap
- **Same questions for both pipelines**: Fair comparison guaranteed
- **Reproducibility**: Random seeds ensure consistent results

## Performance Considerations

### Computational Complexity

- **Multi-view retrieval**: O(V × N × log(N)) where V = views, N = passages
- **Knapsack DP**: O(n × budget) where n = candidates, budget = token limit
- **KG expansion**: O(S × E) where S = seeds, E = edges per seed

### Optimization Strategies

1. **Pool size capping**: Limits knapsack input size
2. **Greedy knapsack**: Faster alternative to DP (use_dp=False)
3. **Batch cross-encoder**: Scores all candidates at once
4. **Caching**: KG loading cached in memory

### Memory Usage

- **KG storage**: ~28K JSON files, ~220K triplets total
- **Vector index**: In-memory FAISS index per question
- **Batch processing**: Processes one question at a time to limit memory

## Troubleshooting

### Common Issues and Solutions

1. **Ollama Connection Error**

   - Check: `ollama serve` is running
   - Verify: Models are pulled (`ollama list`)

2. **Memory Errors**

   - Reduce `MAX_POOL_SIZE` or `token_budget`
   - Use greedy knapsack (`use_dp=False`)

3. **Slow Performance**

   - Use GPU for cross-encoder if available
   - Reduce number of questions for testing
   - Use smaller models

4. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

## Future Improvements

1. **GPU Support**: Add CUDA support for cross-encoder
2. **Caching**: Cache sub-question generation results
3. **Parallel Processing**: Process multiple questions in parallel
4. **Better Token Counting**: Use actual tokenizer instead of approximation
5. **Ablation Studies**: Separate evaluation of each component
