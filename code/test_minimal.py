#!/usr/bin/env python3
"""
Minimal test script for Mac M4 Pro - tests with absolute minimum setup.
This version skips KG extraction and reranking for fastest testing.
"""

import os
import sys
import ujson as json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_response_synthesizer import get_response_synthesizer

# Try to import NaivePostprocessor, but create a simple version if FlagEmbedding fails
try:
    from util.kg_post_processor import NaivePostprocessor
except ImportError as e:
    # If FlagEmbedding import fails, create a minimal NaivePostprocessor
    print(f"⚠️  Warning: Could not import NaivePostprocessor ({e})")
    print("   Creating a minimal version for testing...")
    
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.bridge.pydantic import Field
    from typing import Optional, List
    
    class NaivePostprocessor(BaseNodePostprocessor):
        """Minimal postprocessor for testing without FlagEmbedding."""
        dataset: str = Field(default='hotpotqa')
        
        @classmethod
        def class_name(cls) -> str:
            return "NaivePostprocessor"
        
        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional = None,
        ) -> List[NodeWithScore]:
            # Just return nodes as-is, organized by entity
            entity_order = {}
            sorted_nodes = []
            for i, node in enumerate(nodes):
                node_id = node.node.id_
                if '##' in node_id:
                    ent = node_id.split('##')[0]
                    if ent not in entity_order:
                        entity_order[ent] = len(entity_order)
                    sorted_nodes.append((entity_order[ent], i, node))
            sorted_nodes.sort(key=lambda x: (x[0], x[1]))
            return [node for _, _, node in sorted_nodes]

def test_minimal():
    """
    Minimal test - just answer a question with basic RAG (no KG, no reranking).
    This is the fastest way to test if everything is working.
    """
    
    print("=" * 80)
    print("KG²RAG Minimal Test (Mac M4 Pro)")
    print("=" * 80)
    print("\nThis test uses:")
    print("  ✓ Basic RAG (semantic retrieval + LLM generation)")
    print("  ✗ KG extraction (skipped)")
    print("  ✗ Reranking (skipped)")
    print("  ✗ Graph operations (skipped)")
    print("\nThis is the fastest way to verify your setup works!")
    print("-" * 80)
    
    # Step 1: Check Ollama
    print("\n[1/5] Checking Ollama connection...")
    try:
        Settings.llm = Ollama(model='llama3:8b', request_timeout=60)
        Settings.embed_model = OllamaEmbedding(model_name='mxbai-embed-large')
        print("✓ Ollama models initialized")
    except Exception as e:
        print(f"✗ ERROR: Failed to connect to Ollama: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Check if models are downloaded: ollama list")
        print("  3. Download missing models:")
        print("     ollama pull llama3:8b")
        print("     ollama pull mxbai-embed-large")
        return False
    
    # Step 2: Load a single question from dataset
    print("\n[2/5] Loading dataset...")
    data_path = '../data/hotpotqa/hotpot_dev_distractor_v1.json'
    
    if not os.path.exists(data_path):
        print(f"✗ ERROR: Dataset not found: {data_path}")
        print("\nYou need to download the HotpotQA dataset.")
        print("It's available from: https://hotpotqa.github.io/")
        return False
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if len(data) == 0:
        print("✗ ERROR: Dataset is empty")
        return False
    
    # Get first question
    sample = data[0]
    question = sample['question']
    answer = sample['answer']
    sample_id = sample['_id']
    
    print(f"✓ Loaded dataset ({len(data)} questions)")
    print(f"\nQuestion ID: {sample_id}")
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {answer}")
    
    # Step 3: Create document chunks from context
    print("\n[3/5] Creating document chunks...")
    doc_chunks = []
    
    if 'context' not in sample:
        print("✗ ERROR: Sample doesn't have 'context' field")
        return False
    
    for ctx in sample['context']:
        ent = ctx[0]  # Entity name
        paragraphs = ctx[1]  # List of paragraphs
        
        for i, para in enumerate(paragraphs):
            text = f'{ent}: {para}' if i > 0 else para
            doc_chunk = TextNode(text=text, id_=f'{ent}##{i}')
            doc_chunks.append(doc_chunk)
    
    print(f"✓ Created {len(doc_chunks)} document chunks")
    
    # Step 4: Build vector index
    print("\n[4/5] Building vector index (this may take 10-30 seconds)...")
    try:
        index = VectorStoreIndex(doc_chunks, show_progress=True)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        print("✓ Vector index created")
    except Exception as e:
        print(f"✗ ERROR: Failed to create index: {e}")
        return False
    
    # Step 5: Generate answer
    print("\n[5/5] Generating answer (this may take 5-15 seconds)...")
    print("-" * 80)
    
    try:
        # Create prompt template
        qa_rag_template_str = '''Context information is below.
{context_str}
Give a short factoid answer (as few words as possible) based on the context.
Q: {query_str}
A: '''
        
        qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=qa_rag_prompt_template
        )
        
        # Simple post-processor (just organize chunks)
        naive_pp = NaivePostprocessor(dataset='hotpotqa')
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[naive_pp]
        )
        
        # Query
        response = query_engine.query(question)
        prediction = response.response
        
        # Get supporting facts
        sps = []
        for source_node in response.source_nodes:
            node_id = source_node.node.id_
            if '##' in node_id:
                ent, seq_str = node_id.split('##')
                sps.append([ent, int(seq_str)])
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"\nPredicted Answer: {prediction}")
        print(f"Ground Truth: {answer}")
        print(f"\nSupporting Facts: {sps}")
        print(f"Number of supporting facts: {len(sps)}")
        
        # Check if answer matches
        if prediction.lower().strip() == answer.lower().strip():
            print("\n✓ Answer matches ground truth!")
        else:
            print("\n✗ Answer does not match ground truth")
            print("   (This is normal - the model may give a correct but different answer)")
        
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to generate answer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_minimal()
    
    if success:
        print("\n🎉 Your setup is working! You can now:")
        print("  1. Test with KG extraction: python test_kg_extraction.py")
        print("  2. Test single question with full pipeline: python test_single_question.py")
        print("  3. Run full dataset: python kg_rag_distractor.py")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        sys.exit(1)

