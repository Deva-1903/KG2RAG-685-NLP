#!/usr/bin/env python3
"""
Sub-Question Generation Module

Generate sub-questions for HotpotQA questions (MuSiQue already has sub-questions).
Uses GPT-4 or similar LLM to decompose multi-hop questions.
"""

import json
from typing import List, Dict
import openai
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings


class SubQuestionGenerator:
    """
    Generate sub-questions for multi-hop questions.
    
    For MuSiQue: Use existing sub-questions from dataset
    For HotpotQA: Generate using LLM
    """
    
    def __init__(
        self,
        use_openai: bool = False,
        openai_api_key: str = None,
        openai_model: str = "gpt-4",
        ollama_model: str = "llama3:8b"
    ):
        self.use_openai = use_openai
        if use_openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            self.openai_model = openai_model
        else:
            # Use Ollama
            Settings.llm = Ollama(model=ollama_model, request_timeout=200)
            self.llm = Settings.llm
    
    def generate_for_hotpotqa(self, question: str) -> List[str]:
        """
        Generate sub-questions for a HotpotQA question.
        
        Args:
            question: Original multi-hop question
        
        Returns:
            List of sub-questions
        """
        prompt = f"""Given the following multi-hop question, decompose it into 2-3 single-hop sub-questions that would help answer it.

Original question: {question}

Generate 2-3 sub-questions, one per line. Each sub-question should be answerable independently and together they should cover all aspects needed to answer the original question.

Sub-questions:"""
        
        if self.use_openai:
            try:
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that decomposes multi-hop questions into single-hop sub-questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                result = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._fallback_generation(question)
        else:
            try:
                result = str(self.llm.complete(prompt))
            except Exception as e:
                print(f"Ollama error: {e}")
                return self._fallback_generation(question)
        
        # Parse sub-questions (one per line, numbered or not)
        sub_questions = []
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (e.g., "1. ", "Q1: ", etc.)
            line = line.lstrip('0123456789. ')
            if line.startswith('Q'):
                line = line.split(':', 1)[-1].strip()
            
            if line and len(line) > 10:  # Filter out very short lines
                sub_questions.append(line)
        
        # Ensure we have at least 2 sub-questions
        if len(sub_questions) < 2:
            return self._fallback_generation(question)
        
        return sub_questions[:3]  # Return up to 3
    
    def _fallback_generation(self, question: str) -> List[str]:
        """
        Fallback: Simple heuristic-based decomposition.
        
        Args:
            question: Original question
        
        Returns:
            List of sub-questions (heuristic)
        """
        # Simple heuristic: split on "and", "or", etc.
        question_lower = question.lower()
        
        sub_questions = []
        
        # Check for comparison questions
        if " and " in question_lower or " or " in question_lower:
            # Try to split on "and"/"or"
            parts = question.split(" and ")
            if len(parts) == 1:
                parts = question.split(" or ")
            
            if len(parts) >= 2:
                # Create sub-questions for each part
                for i, part in enumerate(parts[:2]):
                    # Try to create a question from the part
                    if "?" not in part:
                        part = part.strip() + "?"
                    sub_questions.append(part.strip())
        
        # If no good split, create generic sub-questions
        if len(sub_questions) < 2:
            # Generic: "What is X?" and "What is Y?" for "X and Y" questions
            words = question.split()
            if len(words) > 5:
                mid = len(words) // 2
                sub_q1 = " ".join(words[:mid]) + "?"
                sub_q2 = " ".join(words[mid:]) + "?"
                sub_questions = [sub_q1, sub_q2]
            else:
                # Last resort: return question as single sub-question
                sub_questions = [question]
        
        return sub_questions[:3]
    
    def get_subquestions(
        self,
        question: str,
        dataset: str,
        sample: Dict = None
    ) -> List[str]:
        """
        Get sub-questions for a question.
        
        Args:
            question: Original question
            dataset: "hotpotqa" or "musique"
            sample: Full sample (for MuSiQue, contains sub-questions)
        
        Returns:
            List of sub-questions
        """
        if dataset == "musique":
            # MuSiQue has sub-questions in the dataset
            if sample and "decomposition" in sample:
                sub_questions = sample["decomposition"].get("questions", [])
                if sub_questions:
                    return sub_questions
            
            # Fallback: generate if not found
            return self.generate_for_hotpotqa(question)
        
        elif dataset == "hotpotqa":
            # Generate sub-questions for HotpotQA
            return self.generate_for_hotpotqa(question)
        
        else:
            # Unknown dataset: generate
            return self.generate_for_hotpotqa(question)


def load_subquestions_cache(cache_path: str) -> Dict[str, List[str]]:
    """Load cached sub-questions from file."""
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_subquestions_cache(cache: Dict[str, List[str]], cache_path: str):
    """Save sub-questions to cache file."""
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
