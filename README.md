# LLM Knowledge Assistant (RAG-Only)

Beginner-friendly Python RAG project that builds a local FAISS index from StackOverflow Q&A and answers questions with grounded citations.

Pipeline:

`Raw Q/A -> Chunking -> Embeddings -> FAISS Retrieval -> LLM Answer (with citations)`

## Project Structure

```text
llm_ka/
├─ src/
│  ├─ __init__.py
│  ├─ build_index.py
│  ├─ config.py
│  └─ rag.py
├─ scripts/
│  └─ download_stackoverflow.py
└─ data/
   ├─ stackoverflow/
   │  ├─ Questions.csv
   │  ├─ Answers.csv
   │  └─ Tags.csv
   └─ index/
      ├─ faiss.index
      ├─ metadata.pkl
      └─ chunks.pkl
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_stackoverflow.py
```

## Environment Variables

Create `.env` in project root.

### Option A: OpenAI

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

### Option B: OpenAI-compatible providers (Groq/OpenRouter/etc.)

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://your-provider-openai-compatible-endpoint
OPENAI_MODEL=your_model_name
```

### Optional: Hugging Face token (for embedding model download)

```env
HF_TOKEN=your_huggingface_read_token
```

## Build Index (RAG Preparation)

```bash
python -m src.build_index
```

What it does:
- Loads and joins `Questions.csv` + `Answers.csv`
- Creates question and answer chunks
- Generates embeddings with `all-MiniLM-L6-v2`
- Builds FAISS index
- Uses first 5000 joined rows (`load_data().head(5000)`)
- Saves aligned artifacts for retrieval:
  - `data/index/faiss.index`
  - `data/index/metadata.pkl`
  - `data/index/chunks.pkl`

## Run Grounded QA

```bash
python -m src.rag
```

What it does:
- Loads FAISS index + metadata + chunks
- Retrieves relevant chunks for each query
- Applies per-question deduplication before generation (keeps top chunks per `question_id`)
- Sends only retrieved context to the LLM
- Returns grounded answer + citations
- Falls back to:
  - `Answer not found in the provided documentation.`
  when context is insufficient/invalid

### Current Retrieval Defaults

These are the active defaults in `src/rag.py` (`RETRIEVAL_DEFAULTS`):

```python
{
  "k": 24,
  "max_distance": 0.3,
  "relative_margin": 0.25,
  "max_chunks_per_qid": 4,
  "answers_only": True,
  "min_chunk_chars": 80,
}
```

You can tune retrieval in one place by editing `RETRIEVAL_DEFAULTS`.

## Optional Debug Flags

```bash
export RAG_DEBUG=1
export STRICT_EVIDENCE_QUOTES=1
python -m src.rag
```

- `RAG_DEBUG=1`: prints raw model output
- `STRICT_EVIDENCE_QUOTES=1`: enforces quote matching against retrieved context

### Retrieval Debug Mode

Use these flags to inspect how each retrieval filter affects candidates.

```bash
export RAG_RETRIEVE_DEBUG=1
python -m src.rag
```

- `RAG_RETRIEVE_DEBUG=1`: prints retrieval summary counts (raw top-k, passed filters, rejected-by-reason, final results)

Verbose candidate-by-candidate decisions:

```bash
export RAG_RETRIEVE_DEBUG=1
export RAG_RETRIEVE_DEBUG_VERBOSE=1
python -m src.rag
```

- `RAG_RETRIEVE_DEBUG_VERBOSE=1`: prints decision reason for each inspected candidate

## Data Notes

Main relationships:
- `Questions.Id -> Answers.ParentId`
- `Questions.Id -> Tags.Id`

## Common Issues

1. `ModuleNotFoundError: No module named 'src.config'`
- Ensure `src/config.py` exists and run from repo root with `python -m src.rag`.

2. Embedding model download errors
- Ensure internet access or pre-download `sentence-transformers/all-MiniLM-L6-v2`.

3. Missing API key
- Set `OPENAI_API_KEY` in `.env`.
