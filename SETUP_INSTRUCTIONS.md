# Setup Instructions for Teammates

## Quick Setup

1. **Clone the repository** (if not already done):

   ```bash
   git clone <repo-url>
   cd KG2RAG-main
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy English model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up Ollama** (if not already installed):

   - Install Ollama from https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull llama3:8b
     ollama pull mxbai-embed-large
     ```

6. **Verify setup**:
   ```bash
   cd code
   python verify_setup.py  # If this script exists
   ```

## Required Models

### Ollama Models (Required)

- `llama3:8b` - For LLM inference
- `mxbai-embed-large` - For embeddings

### Reranker Model (Optional - has HuggingFace fallback)

- Local path: `../model/bge-reranker-large`
- Or will automatically use `BAAI/bge-reranker-large` from HuggingFace

## Data Requirements

Ensure you have:

- `../data/hotpotqa/hotpot_dev_distractor_v1.json` - Dataset file
- `../data/hotpotqa/kgs/extract_subkgs/` - Knowledge graphs directory

## Troubleshooting

### Issue: `ImportError: cannot import name 'FlagEmbedding'`

**Solution**: Make sure you have the correct versions:

```bash
pip install FlagEmbedding==1.3.4 transformers>=4.57.3 huggingface-hub>=0.36.0
```

### Issue: `spacy.errors.OsError: Can't find model 'en_core_web_sm'`

**Solution**: Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Issue: `ConnectionError: Failed to connect to Ollama`

**Solution**:

- Make sure Ollama is running: `ollama serve`
- Or start it in the background

### Issue: `huggingface_hub.errors.HFValidationError`

**Solution**: Update huggingface-hub:

```bash
pip install --upgrade huggingface-hub>=0.36.0
```

## Python Version

- **Recommended**: Python 3.8+
- **Tested**: Python 3.10, 3.11, 3.13

## Optional Dependencies

### OpenAI (for sub-question generation)

Only needed if you want to use OpenAI API instead of Ollama:

```bash
pip install openai>=1.0.0
```

Then set environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Verification

After setup, test with a small run:

```bash
cd code
python run_experiments.py --num_questions 10 --first
```

This should complete successfully if everything is set up correctly.
