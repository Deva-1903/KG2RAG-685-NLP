#!/usr/bin/env python3
"""
Setup verification script for Mac M4 Pro.
Run this to check if everything is set up correctly.
"""

import os
import sys
import subprocess

def check_ollama():
    """Check if Ollama is installed and running."""
    print("\n[1/6] Checking Ollama...")
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            models = result.stdout.strip().split('\n')[1:]  # Skip header
            if models:
                print(f"  Found {len(models)} model(s):")
                for model in models:
                    if model.strip():
                        print(f"    - {model.strip()}")
            else:
                print("  ⚠️  No models found. You need to download models.")
            return True
        else:
            print("✗ Ollama is installed but not responding")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("  Install from: https://ollama.com/download")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Ollama is not responding (timeout)")
        print("  Try: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def check_ollama_models():
    """Check if required Ollama models are downloaded."""
    print("\n[2/6] Checking Ollama models...")
    required_models = ['llama3:8b', 'mxbai-embed-large']
    missing_models = []
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            output = result.stdout.lower()
            for model in required_models:
                if model.lower() in output:
                    print(f"✓ {model} is downloaded")
                else:
                    print(f"✗ {model} is NOT downloaded")
                    missing_models.append(model)
            
            if missing_models:
                print("\n  To download missing models:")
                for model in missing_models:
                    print(f"    ollama pull {model}")
                return False
            return True
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False

def check_python_dependencies():
    """Check if Python dependencies are installed."""
    print("\n[3/6] Checking Python dependencies...")
    
    # Check each package with correct import names
    package_checks = [
        ('FlagEmbedding', 'FlagEmbedding', 'FlagReranker', True),  # Optional for minimal test
        ('llama_index', 'llama_index', None, False),
        ('networkx', 'networkx', None, False),
        ('pandas', 'pandas', None, False),
        ('tqdm', 'tqdm', None, False),
        ('ujson', 'ujson', None, False)
    ]
    
    missing_packages = []
    optional_packages = []
    for display_name, import_name, submodule, is_optional in package_checks:
        try:
            if submodule:
                # For packages with submodules, try importing the submodule
                __import__(import_name)
                mod = __import__(import_name, fromlist=[submodule])
                getattr(mod, submodule)
            else:
                __import__(import_name)
            print(f"✓ {display_name} is installed")
        except (ImportError, AttributeError) as e:
            if is_optional:
                print(f"⚠️  {display_name} has import issues (optional for minimal test)")
                print(f"     Error: {str(e)[:60]}...")
                optional_packages.append(display_name)
            else:
                print(f"✗ {display_name} is NOT installed")
                missing_packages.append(display_name)
    
    if missing_packages:
        print("\n  To install missing packages:")
        print("    cd code")
        print("    pip install -r requirements.txt")
        return False
    
    if optional_packages:
        print(f"\n  Note: {', '.join(optional_packages)} has issues but is optional.")
        print("  You can still run test_minimal.py without it.")
    
    return True

def check_dataset():
    """Check if dataset file exists."""
    print("\n[4/6] Checking dataset...")
    data_path = '../data/hotpotqa/hotpot_dev_distractor_v1.json'
    
    if os.path.exists(data_path):
        size = os.path.getsize(data_path) / (1024 * 1024)  # MB
        print(f"✓ Dataset found: {data_path} ({size:.1f} MB)")
        return True
    else:
        print(f"✗ Dataset not found: {data_path}")
        print("  You need to download the HotpotQA dataset")
        print("  Available from: https://hotpotqa.github.io/")
        return False

def check_kg_files():
    """Check if KG files exist (optional)."""
    print("\n[5/6] Checking KG files (optional)...")
    kg_dir = '../data/hotpotqa/kgs/extract_subkgs'
    
    if os.path.exists(kg_dir):
        kg_files = [f for f in os.listdir(kg_dir) if f.endswith('.json')]
        if kg_files:
            print(f"✓ KG files found: {len(kg_files)} files in {kg_dir}")
            return True
        else:
            print(f"⚠️  KG directory exists but is empty: {kg_dir}")
            print("  You can run preprocessing to extract KGs, or skip for minimal test")
            return None  # Not required
    else:
        print(f"⚠️  KG directory not found: {kg_dir}")
        print("  This is optional - you can test without KGs using test_minimal.py")
        return None  # Not required

def check_reranker():
    """Check if reranker model exists (optional)."""
    print("\n[6/6] Checking reranker model (optional)...")
    reranker_path = '../model/bge-reranker-large'
    
    if os.path.exists(reranker_path):
        print(f"✓ Reranker model found: {reranker_path}")
        return True
    else:
        print(f"⚠️  Reranker model not found: {reranker_path}")
        print("  This is optional - you can test without reranking using test_minimal.py")
        print("  To download:")
        print("    cd model")
        print("    pip install huggingface_hub")
        print("    huggingface-cli download BAAI/bge-reranker-large --local-dir ./bge-reranker-large")
        return None  # Not required

def main():
    print("=" * 80)
    print("KG²RAG Setup Verification (Mac M4 Pro)")
    print("=" * 80)
    
    results = {
        'ollama': check_ollama(),
        'models': check_ollama_models(),
        'dependencies': check_python_dependencies(),
        'dataset': check_dataset(),
        'kg_files': check_kg_files(),
        'reranker': check_reranker()
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    required_checks = ['ollama', 'models', 'dependencies', 'dataset']
    optional_checks = ['kg_files', 'reranker']
    
    all_required = all(results[k] for k in required_checks if k in results)
    
    print("\nRequired components:")
    for check in required_checks:
        status = "✓" if results.get(check) else "✗"
        print(f"  {status} {check}")
    
    print("\nOptional components:")
    for check in optional_checks:
        result = results.get(check)
        if result is True:
            print(f"  ✓ {check}")
        elif result is False:
            print(f"  ✗ {check}")
        else:
            print(f"  ⚠️  {check} (not required for minimal test)")
    
    if all_required:
        print("\n🎉 All required components are set up!")
        print("\nYou can now run:")
        print("  1. Minimal test (fastest): python test_minimal.py")
        print("  2. KG extraction test: python test_kg_extraction.py")
        print("  3. Single question test: python test_single_question.py")
    else:
        print("\n❌ Some required components are missing.")
        print("Please fix the issues above and run this script again.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

