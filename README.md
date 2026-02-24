# LLM Knowledge Assistant (`llm_ka`)

Beginner-friendly project to extract structured metadata from StackOverflow questions using an LLM.

## Learning Roadmap

### Phase 1: Information Extraction
- Prompting
- JSON output
- Schema validation
- Confidence scoring
- Failure handling

### Phase 2: Knowledge Assistant (RAG)
- Chunking
- Embeddings
- Vector database
- Retrieval
- Grounding

### Phase 3: Agents
- LangGraph
- Tool calling
- State
- Memory
- Multi-step reasoning

## Real-World Use Cases

1. Ticket triage and routing (Support / IT / Helpdesk)
2. Search and knowledge-base enrichment (metadata for RAG)
3. Analytics: identify recurring user pain points
4. Auto-tagging and governance for data/docs/code

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
