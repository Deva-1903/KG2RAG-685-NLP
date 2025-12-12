#!/usr/bin/env python3
"""
Test script to test KG extraction on a sample text.
This is the smallest test - just extracts triplets from text.
"""

import sys
import os
from llama_index.llms.ollama import Ollama

def extract_triplets(llm, ctx):
    """Extract knowledge graph triplets from text using LLM."""
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

def test_kg_extraction():
    """Test KG extraction on sample texts."""
    
    print("=" * 80)
    print("KG Extraction Test")
    print("=" * 80)
    
    # Initialize LLM
    print("\n[1/2] Initializing LLM...")
    try:
        llm = Ollama(model='llama3:8b', request_timeout=120)
        print("✓ LLM initialized (Ollama must be running)")
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM: {e}")
        print("Make sure Ollama is running: ollama pull llama3:8b")
        return
    
    # Test texts
    test_texts = [
        "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
        "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role.",
        "Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."
    ]
    
    print("\n[2/2] Extracting triplets from test texts...")
    print("-" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print("Extracting triplets...")
        
        try:
            triplets = extract_triplets(llm, text)
            print(f"✓ Extracted {len(triplets)} triplets:")
            for triplet in triplets:
                print(f"  - {triplet}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == '__main__':
    test_kg_extraction()

