#!/usr/bin/env python3
"""
Test version of kg_rag_distractor.py - tests on first 100 questions only.
Keeps original pipeline intact, just limits the dataset.
"""

import os
import copy
import ujson as json
import argparse
import random
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG')
logging.getLogger('KGRAG').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

def read_data(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    if args.dataset == 'hotpotqa':
        with open(data_path,'r',encoding='utf-8') as f:
            data = json.load(f)
    elif args.dataset == 'musique':
        data = []
        with open(data_path,'r',encoding='utf-8') as f:
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
    Settings.llm = Ollama(model=args.model_name,request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)

def read_kg(args,data):
    if args.dataset=='hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir,f'{ent.replace("/","_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path,'r',encoding='utf-8')as f:
                    subkg = json.load(f)
                    repkg = copy.deepcopy(subkg)
                    if subkg and len(subkg.keys())>0:
                        for seq in subkg.keys():
                            if len(repkg[seq])==0:
                                del repkg[seq]
                        if len(repkg.keys())>0:
                            doc2kg[ent] = repkg
    elif args.dataset=='musique':
        kg_dir = args.kg_dir
        kg_path = os.path.join(kg_dir,'musique_kg_filtered.json')
        with open(kg_path,'r',encoding='utf-8') as f:
            doc2kg = json.load(f)
    print(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg

def write_prediction(args,data,prediction,tested_questions,metadata=None):
    result_path = args.result_path
    output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset=='hotpotqa':
        # Save main results
        with open(result_path,'w',encoding='utf-8') as f:
            json.dump(prediction,f, indent=2)
        
        # Save detailed results with question info
        detailed_path = result_path.replace('.json', '_detailed.json')
        detailed_results = {
            'summary': {
                'total_tested': len(tested_questions),
                'questions_tested': tested_questions
            },
            'results': [],
            'metadata': metadata if metadata else []
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
        
    elif args.dataset=='musique':
        with open(result_path,'w',encoding='utf-8') as f:
            for sample in data:
                sample_id = sample['id']
                sample['predicted_answer'] = prediction['answer'][sample_id]
                sample['predicted_support_idxs'] = prediction['sp'][sample_id]
                sample['predicted_answerable'] = sample['answerable']
                f.write(json.dumps(sample)+'\n')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

def process_sample(args,sample,kg):
    if args.dataset=='hotpotqa':
        sample_id = sample['_id']
    elif args.dataset=='musique':
        sample_id = sample['id']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    sample_question = sample['question']
    sample_answer = sample['answer']

    ents = set()
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    if args.dataset=='hotpotqa':
        ctxs = sample['context']
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                if (ent in kg) and (str(i) in kg[ent]) and (len(kg[ent][str(i)])>0):
                    if ent not in subkg:
                        subkg[ent] = dict()
                    target_kg = kg[ent][str(i)]
                    for triplet in target_kg:
                        h,r,t = triplet
                        if ngram_overlap(h,ent)>=0.90 or ngram_overlap(ent,h)>=0.90:
                            h = ent
                        if ngram_overlap(t,ent)>=0.90 or ngram_overlap(ent,t)>=0.90:
                            t = ent
                        triplet = (h,r,t)
                    subkg[ent][str(i)] = target_kg
                text = f'{ent}: {ctx[1][i]}'
                doc_chunk = TextNode(text=text,id_=f'{ent}##{str(i)}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = text
    elif args.dataset=='musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            seq = ctx['seq']
            text = f'{ent}: {ctx["paragraph_text"]}'
            if (ent in kg) and (str(seq) in kg[ent]) and (len(kg[ent][str(seq)])>0):
                if ent not in subkg:
                    subkg[ent] = dict()
                subkg[ent][f'{str(idx)}##{str(seq)}'] = kg[ent][str(seq)]
            doc_chunk = TextNode(text=text,id_=f'{str(idx)}##{ent}##{str(seq)}')
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f'{str(idx)}##{str(seq)}'] = text

    index = VectorStoreIndex(doc_chunks)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=args.top_k)
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)

    expansion_pp = KGRetrievePostProcessor(dataset=args.dataset,ents=ents,doc2kg=subkg,chunks_index=chunks_index)
    # Handle reranker path: convert relative path to absolute, fallback to HuggingFace if not found
    reranker_path = args.reranker
    if not os.path.isabs(reranker_path):
        # Convert relative path to absolute
        reranker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), reranker_path))
    
    # If local path doesn't exist, use HuggingFace model ID
    if not os.path.exists(reranker_path):
        # Use the standard BGE reranker from HuggingFace
        reranker_path = 'BAAI/bge-reranker-large'
        print(f'Local reranker model not found, using HuggingFace model: {reranker_path}')
    
    bge_reranker = FlagReranker(model_name_or_path=reranker_path,device=3)
    filter_pp = GraphFilterPostProcessor(dataset=args.dataset,use_tpt=args.use_tpt,topk=args.top_k,ents=ents,doc2kg=subkg,chunks_index=chunks_index,reranker=bge_reranker)
    naive_pp = NaivePostprocessor(dataset=args.dataset)
    query_engine = RetrieverQueryEngine(retriever=retriever,response_synthesizer=response_synthesizer,node_postprocessors=[expansion_pp,filter_pp,naive_pp])

    try:
        response = query_engine.query(sample_question)
        prediction = response.response
        if args.dataset=='hotpotqa':
            sps = [[source_node.node.id_.split('##')[0],int(source_node.node.id_.split('##')[1])] for source_node in response.source_nodes]
            sps = [[ent,seq]for ent,seq in sps if (seq>=0)]
        elif args.dataset=='musique':
            sps = [int(source_node.node.id_.split('##')[0]) for source_node in response.source_nodes]
            sps = [idx for idx in sps if (idx>=0)]
    except Exception as e:
        print(f'Sample {sample_id}, Error: {e}')
        prediction = ''
        sps = []
        response = None
    
    # Calculate tokens used (context + answer)
    tokens_used = 0
    try:
        if response and hasattr(response, 'source_nodes'):
            # Count tokens in context (from source nodes)
            context_text = ' '.join([node.node.text for node in response.source_nodes])
            tokens_used = int(len(context_text.split()) * 1.3)  # Approximate
        # Count tokens in answer
        if prediction:
            tokens_used += int(len(prediction.split()) * 1.3)
    except Exception:
        # Fallback: simple word count
        tokens_used = len(prediction.split()) if prediction else 0
    
    return sample_id, prediction, sps, {'tokens_used': tokens_used}

def kgrag_distractor_predict(args,data,kg):
    prediction = {'answer':{},'sp':{}}
    sps_count = 0
    tested_questions = []
    metadata = []
    
    for sample in tqdm(data, desc="Processing questions"):
        sample_id = sample['_id'] if args.dataset == 'hotpotqa' else sample['id']
        tested_questions.append({
            'id': sample_id,
            'question': sample['question'],
            'ground_truth': sample['answer']
        })
        
        sample_id, sample_prediction, sample_sps, sample_metadata = process_sample(args, sample, kg)
        prediction['answer'][sample_id] = sample_prediction
        prediction['sp'][sample_id] = sample_sps
        sps_count += len(sample_sps)
        
        # Store metadata for evaluation
        metadata.append({
            'sample_id': sample_id,
            **sample_metadata
        })
    
    print(f'Average number of supporting facts: {sps_count/len(data)}')
    return prediction, tested_questions, metadata

def main(args):
    data = read_data(args)
    init_model(args)
    kg = read_kg(args,data)
    prediction, tested_questions, metadata = kgrag_distractor_predict(args,data,kg)
    write_prediction(args,data,prediction,tested_questions,metadata)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Questions tested: {len(tested_questions)}")
    print(f"Results saved to: {args.result_path}")
    print(f"Detailed results saved to: {args.result_path.replace('.json', '_detailed.json')}")
    print("\nFirst 5 questions tested:")
    for i, q in enumerate(tested_questions[:5], 1):
        print(f"  {i}. ID: {q['id']}")
        print(f"     Q: {q['question'][:60]}...")
        print(f"     Answer: {q['ground_truth']}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # hotpotqa distractor
    parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    parser.add_argument('--data_path',type=str,default='../data/hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    parser.add_argument('--kg_dir',type=str,default='../data/hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    parser.add_argument('--result_path',type=str,default='../output/hotpot/hotpot_dev_distractor_v1_kgrag_100.json',help='Path to the result file')
    parser.add_argument('--num_questions',type=int,default=100,help='Number of questions to test (default: 100)')
    parser.add_argument('--random_sample',action='store_true',help='Randomly sample questions instead of taking first N')
    parser.add_argument('--seed',type=int,default=42,help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--embed_model_name',type=str,default='mxbai-embed-large',help='Ollama embedding model name for indexing')
    parser.add_argument('--model_name',type=str,default='llama3:8b',help='Ollama model name')
    parser.add_argument('--reranker',type=str,default='../model/bge-reranker-large',help='Path of the reranker model')
    parser.add_argument('--top_k',type=int,default=10,help='Top k similar documents')

    args = parser.parse_args()

    main(args)
