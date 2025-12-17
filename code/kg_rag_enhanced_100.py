#!/usr/bin/env python3
"""
Experimental KG²RAG Pipeline - Multi-View Retrieval + Knapsack Selection

This is an experimental alternative approach to the original KG²RAG:
1. Multi-view seed retrieval using sub-questions (instead of single-view)
2. Token-budgeted knapsack selection (instead of graph ordering + top-M)

Test version: Tests on first 100 questions from HotpotQA
"""

import os
import copy
import ujson as json
import argparse
import random
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
import numpy as np

# Original KG²RAG imports
from FlagEmbedding import FlagReranker
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor, KGRetrievePostProcessor, ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

# New components
from multi_view_retrieval import MultiViewRetriever
from knapsack_selection import KnapsackSelector, extract_entities, count_tokens
from subquestion_generation import SubQuestionGenerator

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG_ENHANCED')
logging.getLogger('KGRAG_ENHANCED').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)


def read_data(args):
    """Load and limit dataset to first 100 questions."""
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    
    if args.dataset == 'hotpotqa':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif args.dataset == 'musique':
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    # Limit to N questions for testing
    num_questions = getattr(args, 'num_questions', 100)
    random_sample = getattr(args, 'random_sample', False)
    seed = getattr(args, 'seed', None)
    start_idx = getattr(args, 'start_idx', None)  # For batch processing
    
    if num_questions and num_questions > 0:
        original_count = len(data)
        if start_idx is not None:
            # Batch processing: get specific range
            end_idx = min(start_idx + num_questions, len(data))
            data = data[start_idx:end_idx]
            print(f'Using batch range: questions {start_idx} to {end_idx} (from {original_count} total)')
        elif random_sample:
            # Random sampling
            if seed is not None:
                random.seed(seed)
                print(f'Using random seed: {seed}')
            data = random.sample(data, min(num_questions, len(data)))
            print(f'Randomly sampled {len(data)} questions (from {original_count} total)')
        else:
            # First N questions (original behavior)
            data = data[:num_questions]
            print(f'Limited dataset to first {len(data)} questions (from {original_count} total)')
    
    return data


def init_model(args):
    """Initialize Ollama LLM and embeddings."""
    Settings.llm = Ollama(model=args.model_name, request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)


def read_kg(args, data):
    """Load pre-extracted knowledge graphs."""
    if args.dataset == 'hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path, 'r', encoding='utf-8') as f:
                    subkg = json.load(f)
                    repkg = copy.deepcopy(subkg)
                    if subkg and len(subkg.keys()) > 0:
                        for seq in subkg.keys():
                            if len(repkg[seq]) == 0:
                                del repkg[seq]
                        if len(repkg.keys()) > 0:
                            doc2kg[ent] = repkg
    elif args.dataset == 'musique':
        kg_dir = args.kg_dir
        kg_path = os.path.join(kg_dir, 'musique_kg_filtered.json')
        with open(kg_path, 'r', encoding='utf-8') as f:
            doc2kg = json.load(f)
    
    print(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg


def build_document_index(sample, dataset='hotpotqa'):
    """
    Build VectorStoreIndex from sample's context documents.
    Returns index and chunks_index for reference.
    """
    doc_chunks = []
    chunks_index = dict()
    
    if dataset == 'hotpotqa':
        ctxs = sample['context']
        for ctx in ctxs:
            ent = ctx[0]
            if ent not in chunks_index:
                chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                text = f'{ent}: {ctx[1][i]}'
                doc_chunk = TextNode(text=text, id_=f'{ent}##{str(i)}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = text
    elif dataset == 'musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            seq = ctx['seq']
            text = f'{ent}: {ctx["paragraph_text"]}'
            if ent not in chunks_index:
                chunks_index[ent] = {}
            doc_chunk = TextNode(text=text, id_=f'{str(idx)}##{ent}##{str(seq)}')
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f'{str(idx)}##{str(seq)}'] = text
    
    index = VectorStoreIndex(doc_chunks)
    return index, chunks_index


def get_gold_supporting_facts(sample, dataset='hotpotqa'):
    """Extract gold supporting facts for evaluation."""
    if dataset == 'hotpotqa':
        # Gold supporting facts are in sample['supporting_facts']
        # Format: [["Entity", seq], ...]
        return sample.get('supporting_facts', [])
    elif dataset == 'musique':
        # MuSiQue format may differ
        return sample.get('supporting_facts', [])
    return []


def process_sample_enhanced(args, sample, kg, subquestion_generator):
    """
    Process a single sample using enhanced pipeline:
    1. Generate/get sub-questions
    2. Multi-view retrieval for seeds
    3. KG expansion (1-hop)
    4. Knapsack selection
    5. Answer generation
    """
    if args.dataset == 'hotpotqa':
        sample_id = sample['_id']
    elif args.dataset == 'musique':
        sample_id = sample['id']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    sample_question = sample['question']
    sample_answer = sample['answer']
    
    # Build document index
    index, chunks_index = build_document_index(sample, args.dataset)
    
    # Step 1: Generate/get sub-questions
    sub_questions = subquestion_generator.get_subquestions(
        question=sample_question,
        dataset=args.dataset,
        sample=sample
    )
    print(f'Sample {sample_id}: Generated {len(sub_questions)} sub-questions')
    
    # Step 2: Multi-view seed retrieval
    multi_view_retriever = MultiViewRetriever(
        index=index,
        fusion_method=args.fusion_method,  # "rrf" or "mean"
        top_n_per_view=args.top_n_per_view,
        final_top_k=args.seed_top_k,
        device='cpu'  # Adjust if GPU available
    )
    
    seed_passages = multi_view_retriever.retrieve_seeds(
        main_question=sample_question,
        sub_questions=sub_questions
    )
    print(f'Sample {sample_id}: Retrieved {len(seed_passages)} seed passages')
    
    # Step 3: KG expansion (1-hop) - use original KG expansion logic
    ents = set()
    subkg = dict()
    
    if args.dataset == 'hotpotqa':
        ctxs = sample['context']
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            for i in range(len(ctx[1])):
                if (ent in kg) and (str(i) in kg[ent]) and (len(kg[ent][str(i)]) > 0):
                    if ent not in subkg:
                        subkg[ent] = dict()
                    target_kg = kg[ent][str(i)]
                    for triplet in target_kg:
                        h, r, t = triplet
                        if ngram_overlap(h, ent) >= 0.90 or ngram_overlap(ent, h) >= 0.90:
                            h = ent
                        if ngram_overlap(t, ent) >= 0.90 or ngram_overlap(ent, t) >= 0.90:
                            t = ent
                        triplet = (h, r, t)
                    subkg[ent][str(i)] = target_kg
    elif args.dataset == 'musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            ents.add(ent)
            seq = ctx['seq']
            if (ent in kg) and (str(seq) in kg[ent]) and (len(kg[ent][str(seq)]) > 0):
                if ent not in subkg:
                    subkg[ent] = dict()
                subkg[ent][f'{str(idx)}##{str(seq)}'] = kg[ent][str(seq)]
    
    # Expand from seeds using KG (1-hop expansion)
    expansion_pp = KGRetrievePostProcessor(
        dataset=args.dataset,
        ents=ents,
        doc2kg=subkg,
        chunks_index=chunks_index
    )

    # Build NodeWithScore list for seed passages
    # seed_passages format: [(doc_id, score, text), ...]
    seed_nodes = []
    seed_doc_ids = set()
    for doc_id, score, text in seed_passages:
        if doc_id in seed_doc_ids:
            continue  # Skip duplicates
        seed_doc_ids.add(doc_id)
        # Create node directly from seed passage (text already available from multi-view retrieval)
        node = TextNode(text=text, id_=doc_id)
        seed_nodes.append(NodeWithScore(node=node, score=float(score)))
    
    print(f'Sample {sample_id}: {len(seed_nodes)} seed nodes prepared for KG expansion')

    # Apply KG expansion to seed nodes to get expanded candidates
    try:
        query_bundle = QueryBundle(query_str=sample_question)
        expanded_nodes = expansion_pp._postprocess_nodes(seed_nodes, query_bundle)
        print(f'Sample {sample_id}: KG expansion added {len(expanded_nodes) - len(seed_nodes)} passages')
    except Exception as e:
        print(f'Sample {sample_id}: KG expansion failed: {e}, using seeds only')
        # Fallback: if expansion fails, just use seed nodes
        expanded_nodes = seed_nodes

    # Build candidate pool from seeds + expanded (dedup)
    candidates = []
    seen_ids = set()
    for nws in expanded_nodes:
        doc_id = nws.node.id_
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        text = nws.node.text
        candidates.append((doc_id, text))
    
    # Cap pool size to ≤ 60 chunks as per proposal
    MAX_POOL_SIZE = 60
    if len(candidates) > MAX_POOL_SIZE:
        # Keep seeds first, then top expanded by score
        seed_ids = {doc_id for doc_id, _, _ in seed_passages}
        seed_candidates = [(doc_id, text) for doc_id, text in candidates if doc_id in seed_ids]
        expanded_candidates = [(doc_id, text) for doc_id, text in candidates if doc_id not in seed_ids]
        
        # If we have seeds, prioritize them; otherwise take top by order
        if seed_candidates:
            remaining_slots = MAX_POOL_SIZE - len(seed_candidates)
            candidates = seed_candidates + expanded_candidates[:max(0, remaining_slots)]
        else:
            candidates = candidates[:MAX_POOL_SIZE]
        
        print(f'Sample {sample_id}: Candidate pool capped from {len(expanded_nodes)} to {len(candidates)} (max {MAX_POOL_SIZE})')
    else:
        print(f'Sample {sample_id}: Candidate pool size: {len(candidates)} (seeds + KG-expanded)')

    # Step 5: Knapsack selection
    # Extract entities from question
    question_entities = extract_entities(sample_question)
    subquestion_keywords = set()
    for sq in sub_questions:
        subquestion_keywords.update(sq.lower().split())

    knapsack_selector = KnapsackSelector(
        token_budget=args.token_budget,
        use_dp=args.use_dp_knapsack,
        device='cpu'
    )

    selected_doc_ids, selected_values, total_value = knapsack_selector.select_evidence(
        question=sample_question,
        candidates=candidates,
        question_entities=question_entities,
        subquestion_keywords=subquestion_keywords
    )
    
    print(f'Sample {sample_id}: Selected {len(selected_doc_ids)} passages via knapsack')
    
    # Step 6: Answer generation with selected passages
    # Build selected nodes from candidates (we already have text from candidates)
    selected_nodes = []
    candidate_dict = {doc_id: text for doc_id, text in candidates}
    
    for doc_id in selected_doc_ids:
        if doc_id in candidate_dict:
            text = candidate_dict[doc_id]
            node = TextNode(text=text, id_=doc_id)
            selected_nodes.append(node)
        else:
            # Fallback: try to get from chunks_index
            if args.dataset == 'hotpotqa':
                parts = doc_id.split('##')
                if len(parts) == 2:
                    ent, seq_str = parts
                    if ent in chunks_index and seq_str in chunks_index[ent]:
                        text = chunks_index[ent][seq_str]
                        node = TextNode(text=text, id_=doc_id)
                        selected_nodes.append(node)
            elif args.dataset == 'musique':
                parts = doc_id.split('##')
                if len(parts) >= 2:
                    ent = parts[1] if len(parts) > 1 else parts[0]
                    idx_seq = doc_id
                    if ent in chunks_index and idx_seq in chunks_index[ent]:
                        text = chunks_index[ent][idx_seq]
                        node = TextNode(text=text, id_=doc_id)
                        selected_nodes.append(node)
    
    if not selected_nodes:
        print(f'Sample {sample_id}: Warning - No selected nodes found, using fallback')
        # Fallback: use original index with top-k
        selected_index = index
        selected_retriever = VectorIndexRetriever(index=selected_index, similarity_top_k=min(args.top_k, len(selected_doc_ids)))
    else:
        selected_index = VectorStoreIndex(selected_nodes)
        selected_retriever = VectorIndexRetriever(index=selected_index, similarity_top_k=len(selected_nodes))
    
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_rag_prompt_template
    )
    
    naive_pp = NaivePostprocessor(dataset=args.dataset)
    query_engine = RetrieverQueryEngine(
        retriever=selected_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[naive_pp]
    )
    
    try:
        response = query_engine.query(sample_question)
        prediction = response.response
        
        if args.dataset == 'hotpotqa':
            sps = [
                [source_node.node.id_.split('##')[0], int(source_node.node.id_.split('##')[1])]
                for source_node in response.source_nodes
            ]
            sps = [[ent, seq] for ent, seq in sps if (seq >= 0)]
        elif args.dataset == 'musique':
            sps = [int(source_node.node.id_.split('##')[0]) for source_node in response.source_nodes]
            sps = [idx for idx in sps if (idx >= 0)]
    except Exception as e:
        print(f'Sample {sample_id}, Error: {e}')
        prediction = ''
        sps = []
    
    # Calculate tokens used (context + answer)
    tokens_used = 0
    try:
        from knapsack_selection import count_tokens
        # Count tokens in selected passages (context)
        for doc_id in selected_doc_ids:
            if doc_id in candidate_dict:
                tokens_used += count_tokens(candidate_dict[doc_id])
        # Count tokens in answer
        if prediction:
            tokens_used += count_tokens(prediction)
    except Exception:
        # Fallback: approximate token count
        total_text = ' '.join([candidate_dict.get(doc_id, '') for doc_id in selected_doc_ids])
        tokens_used = int(len(total_text.split()) * 1.3) + len(prediction.split()) if prediction else 0
    
    return sample_id, prediction, sps, {
        'sub_questions': sub_questions,
        'num_seeds': len(seed_passages),
        'num_selected': len(selected_doc_ids),
        'total_value': total_value,
        'tokens_used': tokens_used
    }


def kgrag_enhanced_predict(args, data, kg):
    """Run enhanced pipeline on all samples."""
    prediction = {'answer': {}, 'sp': {}}
    tested_questions = []
    metadata = []
    
    # Initialize sub-question generator
    subquestion_generator = SubQuestionGenerator(
        use_openai=args.use_openai,
        openai_api_key=args.openai_api_key,
        ollama_model=args.model_name
    )
    
    for sample in tqdm(data, desc="Processing questions"):
        sample_id = sample['_id'] if args.dataset == 'hotpotqa' else sample['id']
        tested_questions.append({
            'id': sample_id,
            'question': sample['question'],
            'ground_truth': sample['answer']
        })
        
        sample_id, sample_prediction, sample_sps, sample_metadata = process_sample_enhanced(
            args, sample, kg, subquestion_generator
        )
        
        prediction['answer'][sample_id] = sample_prediction
        prediction['sp'][sample_id] = sample_sps
        metadata.append({
            'sample_id': sample_id,
            **sample_metadata
        })
    
    return prediction, tested_questions, metadata


def write_prediction(args, data, prediction, tested_questions, metadata):
    """Write predictions and detailed results."""
    result_path = args.result_path
    output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == 'hotpotqa':
        # Save main results
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, indent=2)
        
        # Save detailed results
        detailed_path = result_path.replace('.json', '_detailed.json')
        detailed_results = {
            'summary': {
                'total_tested': len(tested_questions),
                'questions_tested': tested_questions,
                'pipeline': 'experimental_kg2rag',
                'components': ['multi_view_retrieval', 'knapsack_selection']
            },
            'results': [],
            'metadata': metadata
        }
        
        for sample in data:
            sample_id = sample['_id'] if args.dataset == 'hotpotqa' else sample['id']
            detailed_results['results'].append({
                'question_id': sample_id,
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'prediction': prediction['answer'].get(sample_id, ''),
                'supporting_facts': prediction['sp'].get(sample_id, [])
            })
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f'Prediction written to {result_path}')
        print(f'Detailed results written to {detailed_path}')
    
    elif args.dataset == 'musique':
        with open(result_path, 'w', encoding='utf-8') as f:
            for sample in data:
                sample_id = sample['id']
                sample['predicted_answer'] = prediction['answer'][sample_id]
                sample['predicted_support_idxs'] = prediction['sp'][sample_id]
                sample['predicted_answerable'] = sample['answerable']
                f.write(json.dumps(sample) + '\n')


def main(args):
    """Main pipeline execution."""
    data = read_data(args)
    init_model(args)
    kg = read_kg(args, data)
    prediction, tested_questions, metadata = kgrag_enhanced_predict(args, data, kg)
    write_prediction(args, data, prediction, tested_questions, metadata)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENTAL KG²RAG TEST SUMMARY")
    print("=" * 80)
    print(f"Questions tested: {len(tested_questions)}")
    print(f"Results saved to: {args.result_path}")
    print(f"Detailed results saved to: {args.result_path.replace('.json', '_detailed.json')}")
    print("\nPipeline components:")
    print("  - Multi-view seed retrieval")
    print("  - KG expansion (1-hop)")
    print("  - Knapsack selection")
    print("\nFirst 5 questions tested:")
    for i, q in enumerate(tested_questions[:5], 1):
        print(f"  {i}. ID: {q['id']}")
        print(f"     Q: {q['question'][:60]}...")
        print(f"     Answer: {q['ground_truth']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental KG²RAG Pipeline: Multi-View Retrieval + Knapsack Selection')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='hotpotqa', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/hotpot_dev_distractor_v1.json', help='Path to the data file')
    parser.add_argument('--kg_dir', type=str, default='../data/hotpotqa/kgs/extract_subkgs', help='Directory of the KGs')
    parser.add_argument('--result_path', type=str, default='../output/hotpot/hotpot_dev_distractor_v1_kgrag_experimental_100.json', help='Path to the result file')
    parser.add_argument('--num_questions', type=int, default=100, help='Number of questions to test (default: 100)')
    parser.add_argument('--random_sample', action='store_true', help='Randomly sample questions instead of taking first N')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index for batch processing (for non-overlapping batches)')
    
    # Model arguments
    parser.add_argument('--embed_model_name', type=str, default='mxbai-embed-large', help='Ollama embedding model name')
    parser.add_argument('--model_name', type=str, default='llama3:8b', help='Ollama model name')
    parser.add_argument('--top_k', type=int, default=10, help='Top k similar documents for retrieval')
    
    # Multi-view retrieval arguments
    parser.add_argument('--fusion_method', type=str, default='rrf', choices=['rrf', 'mean'], help='Fusion method: rrf or mean')
    parser.add_argument('--top_n_per_view', type=int, default=10, help='Top N passages per sub-question view')
    parser.add_argument('--seed_top_k', type=int, default=8, help='Final number of seed passages after MMR')
    
    # Knapsack selection arguments
    parser.add_argument('--token_budget', type=int, default=2048, help='Token budget for knapsack selection')
    parser.add_argument('--use_dp_knapsack', action='store_true', help='Use DP (exact) knapsack solver, otherwise greedy')
    
    # Sub-question generation arguments
    parser.add_argument('--use_openai', action='store_true', help='Use OpenAI for sub-question generation')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key')
    
    args = parser.parse_args()
    
    main(args)
