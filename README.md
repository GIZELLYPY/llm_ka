# LLM Knowledge Assistant
Texto Bruto -> Chunking -> Embeddings -> Banco de Vetores + Índice -> Busca de Similaridade -> LLM

## Project Structure

```text
llm_ka/
├─ src/
│  ├─ config.py
│  ├─ schema.py
│  ├─ extractor.py
│  ├─ load_data.py
│  ├─ retry.py
│  ├─ run_extract.py
│  └─ eval.py
├─ scripts/
│  └─ download_stackoverflow.py
└─ data/
   └─ stackoverflow/
      ├─ Questions.csv
      ├─ Answers.csv
      └─ Tags.csv
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

### Option B: Groq (OpenAI-compatible endpoint)

```env
OPENAI_API_KEY=your_groq_api_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

### Option C: OpenRouter (OpenAI-compatible endpoint)

```env
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

### Hugging Face Token (for embedding model downloads)

```env
HF_TOKEN=your_huggingface_read_token
```

`HF_TOKEN` is used when downloading `sentence-transformers/all-MiniLM-L6-v2`.
Without it, downloads still work but with stricter rate limits.
For create a token (just read permission): https://huggingface.co/settings/tokens

## Run Extraction

```bash
python3 -m src.run_extract
```

What it does:
- Loads first 20 questions from `data/stackoverflow/Questions.csv`
- Calls LLM extraction with one retry on transient failures
- Writes success records to `data/extracted_questions.jsonl`
- Writes failures to `data/failed_extractions.jsonl`

## Evaluate Results

```bash
python3 -m src.eval
```

## Build FAISS Index (RAG Prep)

```bash
python -m src.build_index
```

What it does:
- Loads and joins `Questions.csv` + `Answers.csv`
- Builds text chunks (question + answer)
- Creates embeddings with `all-MiniLM-L6-v2`
- Builds and saves FAISS index
- Saves metadata mapping for retrieval

Generated files:
- `data/index/faiss.index`
- `data/index/metadata.pkl`

Expected note in logs:
- `embeddings.position_ids | UNEXPECTED`
- This is usually harmless for this model load path and can be ignored.

## Output Format (`JSONL`)

`JSONL` means one JSON object per line.

Example:

```json
{"question_id": 1, "title": "A", "intent": "how-to"}
{"question_id": 2, "title": "B", "intent": "debugging"}
```

## Data Files

### `Questions.csv`
- `Id`: question ID
- `Title`: question title
- `Body`: full question text (HTML)
- `Score`: upvotes
- `Tags`: tag string (example: `python|pandas|dataframe`)
- `CreationDate`

### `Answers.csv`
- `Id`: answer ID
- `ParentId`: question ID (foreign key to `Questions.Id`)
- `Body`: answer text
- `Score`
- `CreationDate`

### `Tags.csv`
- `Id`: question ID
- `Tag`: single normalized tag (example: `python`, `spark`)

Relationship:
- `Questions.Id` -> `Answers.ParentId`
- `Questions.Id` -> `Tags.Id`
