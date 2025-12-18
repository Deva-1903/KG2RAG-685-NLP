#!/usr/bin/env python3
"""
Multi-View Seed Retrieval Module

Novelty #1: Instead of single-view retrieval, retrieve passages for each sub-question
separately, then fuse using RRF/mean-cosine, rerank with cross-encoder, and apply MMR.

This ensures both hops are represented before KG expansion.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import spacy
from sentence_transformers import CrossEncoder
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex

# Try to load spaCy model, fallback to basic if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Using basic tokenizer.")
    nlp = None


def extract_entities(text: str) -> Set[str]:
    """Extract named entities from text using spaCy."""
    if nlp is None:
        # Fallback: simple word extraction
        return set(text.split())
    
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        entities.add(ent.text.lower())
    return entities


def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple rankings.
    
    Args:
        rankings: List of rankings, each is [(doc_id, score), ...]
        k: RRF constant (default 60)
    
    Returns:
        Dictionary mapping doc_id to fused score
    """
    doc_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, (doc_id, score) in enumerate(ranking, start=1):
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    return dict(doc_scores)


def mean_cosine_fusion(rankings: List[List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    Mean cosine fusion: average scores across views.
    
    Args:
        rankings: List of rankings, each is [(doc_id, score), ...]
    
    Returns:
        Dictionary mapping doc_id to mean score
    """
    doc_scores = defaultdict(list)
    
    for ranking in rankings:
        for doc_id, score in ranking:
            doc_scores[doc_id].append(score)
    
    # Average scores
    return {doc_id: np.mean(scores) for doc_id, scores in doc_scores.items()}


def mmr_diversity_filter(
    candidates: List[Tuple[str, float, str]],  # (doc_id, score, text)
    query: str,
    top_k: int = 8,
    lambda_param: float = 0.5
) -> List[str]:
    """
    Maximal Marginal Relevance (MMR) filter to ensure diversity.
    
    Args:
        candidates: List of (doc_id, score, text) tuples
        query: Original question
        top_k: Number of diverse results to return
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of selected doc_ids
    """
    if len(candidates) <= top_k:
        return [doc_id for doc_id, _, _ in candidates]
    
    # Sort by score
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    selected = []
    remaining = candidates.copy()
    
    # Select first (highest score)
    if remaining:
        selected.append(remaining.pop(0))
    
    # Greedily select next most diverse
    while len(selected) < top_k and remaining:
        best_idx = 0
        best_mmr = -float('inf')
        
        for idx, (doc_id, score, text) in enumerate(remaining):
            # Relevance term
            relevance = score
            
            # Diversity term: max similarity to already selected
            max_sim = 0.0
            for sel_doc_id, sel_score, sel_text in selected:
                # Simple word overlap as similarity
                sel_words = set(sel_text.lower().split())
                cand_words = set(text.lower().split())
                if len(sel_words) > 0:
                    sim = len(sel_words & cand_words) / len(sel_words | cand_words)
                    max_sim = max(max_sim, sim)
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        
        selected.append(remaining.pop(best_idx))
    
    return [doc_id for doc_id, _, _ in selected]


class MultiViewRetriever:
    """
    Multi-view seed retrieval using sub-questions.
    
    For each sub-question:
    1. Retrieve top-N passages
    2. Fuse rankings (RRF or mean-cosine)
    3. Rerank with cross-encoder
    4. Apply MMR for diversity
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion_method: str = "rrf",  # "rrf" or "mean"
        top_n_per_view: int = 10,
        final_top_k: int = 8,
        device: str = "cpu"
    ):
        self.index = index
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=top_n_per_view)
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=device)
        self.fusion_method = fusion_method
        self.top_n_per_view = top_n_per_view
        self.final_top_k = final_top_k
    
    def retrieve_for_subquestion(self, sub_question: str) -> List[Tuple[str, float, str]]:
        """
        Retrieve passages for a single sub-question.
        
        Returns:
            List of (doc_id, score, text) tuples
        """
        # Retrieve using vector search
        nodes = self.retriever.retrieve(sub_question)
        
        results = []
        for node in nodes:
            doc_id = node.node.id_
            score = node.score if hasattr(node, 'score') else 1.0
            text = node.node.text
            results.append((doc_id, score, text))
        
        return results
    
    def fuse_rankings(
        self,
        per_view_rankings: List[List[Tuple[str, float, str]]]
    ) -> List[Tuple[str, float, str]]:
        """
        Fuse rankings from multiple views.
        
        Args:
            per_view_rankings: List of rankings per view, each is [(doc_id, score, text), ...]
        
        Returns:
            Fused ranking as [(doc_id, fused_score, text), ...]
        """
        # Extract just (doc_id, score) for fusion
        rankings_for_fusion = []
        doc_texts = {}  # Store text for each doc_id
        
        for ranking in per_view_rankings:
            view_ranking = []
            for doc_id, score, text in ranking:
                view_ranking.append((doc_id, score))
                doc_texts[doc_id] = text
            rankings_for_fusion.append(view_ranking)
        
        # Apply fusion
        if self.fusion_method == "rrf":
            fused_scores = reciprocal_rank_fusion(rankings_for_fusion)
        elif self.fusion_method == "mean":
            fused_scores = mean_cosine_fusion(rankings_for_fusion)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Reconstruct with text
        fused_ranking = [
            (doc_id, fused_scores[doc_id], doc_texts[doc_id])
            for doc_id in sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        ]
        
        return fused_ranking
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float, str]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Original question
            candidates: List of (doc_id, score, text) tuples
        
        Returns:
            Reranked list
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, text) for _, _, text in candidates]
        
        # Score with cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Combine with doc_ids
        reranked = [
            (doc_id, float(score), text)
            for (doc_id, _, text), score in zip(candidates, scores)
        ]
        
        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def retrieve_seeds(
        self,
        main_question: str,
        sub_questions: List[str]
    ) -> List[Tuple[str, float, str]]:
        """
        Main method: retrieve diverse seed passages using multi-view retrieval.
        
        Args:
            main_question: Original question
            sub_questions: List of sub-questions (from MuSiQue or generated)
        
        Returns:
            List of (doc_id, score, text) for selected seed passages
        """
        # Step 1: Retrieve for each sub-question
        per_view_rankings = []
        for sub_q in sub_questions:
            ranking = self.retrieve_for_subquestion(sub_q)
            per_view_rankings.append(ranking)
        
        # Also retrieve for main question (add as one view)
        main_ranking = self.retrieve_for_subquestion(main_question)
        per_view_rankings.append(main_ranking)
        
        # Step 2: Fuse rankings
        fused_ranking = self.fuse_rankings(per_view_rankings)
        
        # Step 3: Rerank with cross-encoder
        reranked = self.rerank_with_cross_encoder(main_question, fused_ranking)
        
        # Step 4: Apply MMR for diversity
        selected_doc_ids = mmr_diversity_filter(
            reranked,
            main_question,
            top_k=self.final_top_k
        )
        
        # Return selected passages
        selected = [
            (doc_id, score, text)
            for doc_id, score, text in reranked
            if doc_id in selected_doc_ids
        ]
        
        return selected
    
    def compute_hop_coverage(
        self,
        seed_passages: List[Tuple[str, float, str]],
        gold_supporting_facts: List[Tuple[str, int]]  # [(entity, seq), ...]
    ) -> Dict[str, float]:
        """
        Compute hop coverage: whether both hops are present in seeds.
        
        Args:
            seed_passages: Retrieved seed passages
            gold_supporting_facts: Gold supporting facts (entity, sequence pairs)
        
        Returns:
            Dictionary with coverage metrics
        """
        # Extract entities from seed passages
        seed_entities = set()
        for doc_id, _, text in seed_passages:
            # Parse doc_id: "entity##seq" for HotpotQA
            if "##" in doc_id:
                entity = doc_id.split("##")[0]
                seed_entities.add(entity)
        
        # Extract entities from gold facts
        gold_entities = {entity for entity, _ in gold_supporting_facts}
        
        # Compute coverage
        covered_entities = seed_entities & gold_entities
        entity_coverage = len(covered_entities) / len(gold_entities) if gold_entities else 0.0
        
        # Check if both hops are covered (simplified: at least 2 different entities)
        both_hops_covered = len(covered_entities) >= 2
        
        return {
            "entity_coverage": entity_coverage,
            "both_hops_covered": both_hops_covered,
            "covered_entities": len(covered_entities),
            "total_gold_entities": len(gold_entities)
        }
    
    def compute_recall_at_k(
        self,
        seed_passages: List[Tuple[str, float, str]],
        gold_supporting_facts: List[Tuple[str, int]],
        k: int = 8
    ) -> float:
        """
        Compute Recall@k: fraction of gold supporting facts in top-k seeds.
        
        Args:
            seed_passages: Retrieved seed passages (top k)
            gold_supporting_facts: Gold supporting facts
            k: Top k to consider
        
        Returns:
            Recall@k score
        """
        # Take top k
        top_k_seeds = seed_passages[:k]
        
        # Extract (entity, seq) pairs from seeds
        seed_facts = set()
        for doc_id, _, _ in top_k_seeds:
            if "##" in doc_id:
                parts = doc_id.split("##")
                entity = parts[0]
                seq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
                seed_facts.add((entity, seq))
        
        # Convert gold to set
        gold_facts = set(gold_supporting_facts)
        
        # Compute recall
        if not gold_facts:
            return 0.0
        
        intersection = seed_facts & gold_facts
        recall = len(intersection) / len(gold_facts)
        
        return recall
