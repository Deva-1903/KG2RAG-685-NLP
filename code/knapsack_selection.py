#!/usr/bin/env python3
"""
Token-Budgeted Knapsack Selection Module

Novelty #2: Replace KG2RAG's heuristic ordering/top-M selection with a 0-1 knapsack
optimizer that maximizes value (relevance × coverage) under token budget constraints.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import spacy
from sentence_transformers import CrossEncoder

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Using basic tokenizer.")
    nlp = None


def extract_entities(text: str) -> Set[str]:
    """Extract named entities from text."""
    if nlp is None:
        return set(text.split())
    
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        entities.add(ent.text.lower())
    return entities


def count_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens in text.
    
    Args:
        text: Input text
        tokenizer: Optional tokenizer (if None, uses simple word count)
    
    Returns:
        Token count
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    
    # Fallback: approximate token count (words * 1.3)
    words = len(text.split())
    return int(words * 1.3)


def compute_coverage_bonus(
    passage_text: str,
    question_entities: Set[str],
    subquestion_keywords: Set[str] = None  # Kept for backward compatibility but not used
) -> float:
    """
    Compute coverage bonus: how many question entities appear in passage.
    
    As per proposal: coverage(c) = |ent(q) ∩ ent(c)| / |ent(q)|
    
    Args:
        passage_text: Passage text
        question_entities: Entities from question
        subquestion_keywords: (Optional, kept for compatibility) Keywords from sub-questions
    
    Returns:
        Coverage score [0, 1]
    """
    if not question_entities:
        return 0.0
    
    passage_entities = extract_entities(passage_text.lower())
    
    # Entity coverage as per proposal: |ent(q) ∩ ent(c)| / |ent(q)|
    entity_overlap = len(question_entities & passage_entities)
    coverage = entity_overlap / len(question_entities)
    
    return min(coverage, 1.0)


def knapsack_01_dp(
    values: List[float],
    weights: List[int],
    capacity: int
) -> Tuple[List[int], float]:
    """
    Solve 0-1 knapsack problem using dynamic programming.
    
    Args:
        values: Value of each item
        weights: Weight (token count) of each item
        capacity: Maximum capacity (token budget)
    
    Returns:
        Tuple of (selected_indices, total_value)
    """
    n = len(values)
    
    # DP table: dp[i][w] = max value using first i items with capacity w
    dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i-1][w]
            
            # Take item i if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1]
                )
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            # Item i-1 was selected
            selected.append(i-1)
            w -= weights[i-1]
    
    selected.reverse()
    total_value = dp[n][capacity]
    
    return selected, total_value


def knapsack_greedy(
    values: List[float],
    weights: List[int],
    capacity: int
) -> Tuple[List[int], float]:
    """
    Greedy knapsack solver (faster but suboptimal).
    Selects items by value/weight ratio.
    
    Args:
        values: Value of each item
        weights: Weight of each item
        capacity: Maximum capacity
    
    Returns:
        Tuple of (selected_indices, total_value)
    """
    n = len(values)
    
    # Compute value/weight ratios
    ratios = [(values[i] / weights[i] if weights[i] > 0 else 0, i) for i in range(n)]
    ratios.sort(reverse=True)
    
    selected = []
    total_weight = 0
    total_value = 0.0
    
    for ratio, idx in ratios:
        if total_weight + weights[idx] <= capacity:
            selected.append(idx)
            total_weight += weights[idx]
            total_value += values[idx]
    
    return selected, total_value


class KnapsackSelector:
    """
    Token-budgeted selector using 0-1 knapsack optimization.
    
    Value = CE(q, c) × (1 + coverage(c))
    Cost = token_count(c)
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        token_budget: int = 2048,
        use_dp: bool = True,  # Use DP (exact) or greedy (faster)
        device: str = "cpu"
    ):
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=device)
        self.token_budget = token_budget
        self.use_dp = use_dp
    
    def compute_candidate_value(
        self,
        question: str,
        passage_text: str,
        question_entities: Set[str],
        subquestion_keywords: Set[str],
        cross_encoder_score: float = None
    ) -> float:
        """
        Compute value of a candidate passage.
        
        Value = CE(q, c) × (1 + coverage(c))
        
        Args:
            question: Original question
            passage_text: Passage text
            question_entities: Entities from question
            subquestion_keywords: Keywords from sub-questions
            cross_encoder_score: Pre-computed CE score (optional)
        
        Returns:
            Candidate value
        """
        # Compute cross-encoder score if not provided
        if cross_encoder_score is None:
            ce_score = self.cross_encoder.predict([(question, passage_text)])[0]
            ce_score = float(ce_score)
        else:
            ce_score = cross_encoder_score
        
        # Compute coverage bonus
        coverage = compute_coverage_bonus(
            passage_text,
            question_entities,
            subquestion_keywords
        )
        
        # Final value
        value = ce_score * (1.0 + coverage)
        
        return value
    
    def select_evidence(
        self,
        question: str,
        candidates: List[Tuple[str, str]],  # [(doc_id, text), ...]
        question_entities: Set[str],
        subquestion_keywords: Set[str],
        tokenizer=None
    ) -> Tuple[List[str], List[float], float]:
        """
        Select evidence using knapsack optimization.
        
        Args:
            question: Original question
            candidates: List of (doc_id, text) candidate passages
            question_entities: Entities from question
            subquestion_keywords: Keywords from sub-questions
            tokenizer: Optional tokenizer for accurate token counting
        
        Returns:
            Tuple of (selected_doc_ids, selected_values, total_value)
        """
        if not candidates:
            return [], [], 0.0
        
        n = len(candidates)
        
        # Pre-compute values and weights
        values = []
        weights = []
        doc_ids = []
        
        # Batch cross-encoder scoring for efficiency
        pairs = [(question, text) for _, text in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        
        for (doc_id, text), ce_score in zip(candidates, ce_scores):
            # Compute value
            value = self.compute_candidate_value(
                question,
                text,
                question_entities,
                subquestion_keywords,
                cross_encoder_score=float(ce_score)
            )
            
            # Compute weight (token count)
            weight = count_tokens(text, tokenizer)
            
            values.append(value)
            weights.append(weight)
            doc_ids.append(doc_id)
        
        # Solve knapsack
        if self.use_dp:
            selected_indices, total_value = knapsack_01_dp(values, weights, self.token_budget)
        else:
            selected_indices, total_value = knapsack_greedy(values, weights, self.token_budget)
        
        # Return selected
        selected_doc_ids = [doc_ids[i] for i in selected_indices]
        selected_values = [values[i] for i in selected_indices]
        
        return selected_doc_ids, selected_values, total_value
    
    def compare_with_top_m(
        self,
        question: str,
        candidates: List[Tuple[str, str, float]],  # [(doc_id, text, score), ...]
        question_entities: Set[str],
        subquestion_keywords: Set[str],
        top_m: int = 10,
        tokenizer=None
    ) -> Dict[str, any]:
        """
        Compare knapsack selection vs top-M selection at same budget.
        
        Args:
            question: Original question
            candidates: List of (doc_id, text, score) sorted by score
            question_entities: Entities from question
            subquestion_keywords: Keywords from sub-questions
            top_m: Number for top-M selection
            tokenizer: Optional tokenizer
        
        Returns:
            Comparison dictionary
        """
        # Top-M selection
        top_m_candidates = candidates[:top_m]
        top_m_doc_ids = [doc_id for doc_id, _, _ in top_m_candidates]
        top_m_texts = [text for _, text, _ in top_m_candidates]
        top_m_tokens = sum(count_tokens(text, tokenizer) for text in top_m_texts)
        
        # Knapsack selection
        knapsack_candidates = [(doc_id, text) for doc_id, text, _ in candidates]
        knapsack_doc_ids, knapsack_values, knapsack_total_value = self.select_evidence(
            question,
            knapsack_candidates,
            question_entities,
            subquestion_keywords,
            tokenizer
        )
        knapsack_texts = [text for doc_id, text in zip(knapsack_doc_ids, [t for _, t in knapsack_candidates if _ in knapsack_doc_ids])]
        knapsack_tokens = sum(count_tokens(text, tokenizer) for text in knapsack_texts)
        
        return {
            "top_m": {
                "doc_ids": top_m_doc_ids,
                "num_selected": len(top_m_doc_ids),
                "tokens": top_m_tokens,
                "avg_value": np.mean([score for _, _, score in top_m_candidates]) if top_m_candidates else 0.0
            },
            "knapsack": {
                "doc_ids": knapsack_doc_ids,
                "num_selected": len(knapsack_doc_ids),
                "tokens": knapsack_tokens,
                "total_value": knapsack_total_value,
                "avg_value": np.mean(knapsack_values) if knapsack_values else 0.0
            },
            "comparison": {
                "token_efficiency": knapsack_tokens / top_m_tokens if top_m_tokens > 0 else 1.0,
                "value_improvement": (knapsack_total_value - sum(score for _, _, score in top_m_candidates)) / sum(score for _, _, score in top_m_candidates) if top_m_candidates else 0.0
            }
        }
