from pathlib import Path
import json
import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# LLM
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OPENAI_MODEL, OPENAI_BASE_URL


INDEX_DIR = Path("data/index")
NOT_FOUND = "Answer not found in the provided documentation."
STRICT_EVIDENCE_QUOTES = os.getenv("STRICT_EVIDENCE_QUOTES", "0") == "1"

SYSTEM_PROMPT = """
You are an enterprise documentation assistant.

You must answer strictly and only using the provided CONTEXT.
If the answer is missing or ambiguous, return exactly: "Answer not found in the provided documentation."

Return ONLY valid JSON with this exact shape:
{
  "answer": "string",
  "citations": ["question_id_as_string"],
  "evidence": [
    {
      "question_id": "question_id_as_string",
      "quote": "exact short quote copied verbatim from CONTEXT"
    }
  ]
}

Hard rules:
- Do not output any text outside JSON.
- Every evidence.quote must be copied exactly from CONTEXT.
- If answer != "Answer not found in the provided documentation.", include at least 1 evidence item.
- citations must only include question_id values present in CONTEXT.
- Do not use external knowledge.
""".strip()


def load_index():
    """Load FAISS index + metadata + chunks."""
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, metadata, chunks


def retrieve(
    query: str,
    index,
    model,
    chunks,
    metadata,
    k: int = 24,
    max_distance: float = 1.0,
    relative_margin: float = 0.45,
    max_chunks_per_qid: int = 6,
    answers_only: bool = True,
    min_chunk_chars: int = 160,
):
    """Retrieve chunks and apply strict distance gating to reduce irrelevant context."""
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, k)

    valid_pairs = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        dist = float(distances[0][rank])
        if dist <= max_distance:
            valid_pairs.append((rank, idx, dist))

    if not valid_pairs:
        return []

    best_dist = min(d for _, _, d in valid_pairs)

    per_qid_counts = {}
    results = []
    for rank, idx, dist in valid_pairs:
        meta = metadata[idx]
        qid = str(meta["question_id"])
        mtype = str(meta.get("type", ""))
        chunk_text = str(chunks[idx])



        if dist >  best_dist  + relative_margin:
            continue


        if answers_only and mtype != "answer":
            continue


        if len(chunk_text.strip()) < min_chunk_chars:
            continue


        count = per_qid_counts.get(qid, 0)
        if count >= max_chunks_per_qid:
            continue
        per_qid_counts[qid] = count + 1

        results.append({
            "rank": rank + 1,
            "distance": dist,
            "chunk": chunk_text,
            "meta": meta,
        })


    results.sort(key=lambda r: (r["distance"]))
    return results


def _safe_json_load(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:

        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
        raise


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _validate_response(payload, context: str, allowed_qids: set[str]):
    if not isinstance(payload, dict):
        return False

    answer = payload.get("answer")
    citations = payload.get("citations", [])
    evidence = payload.get("evidence", [])

    if not isinstance(answer, str):
        return False
    if not isinstance(citations, list) or not isinstance(evidence, list):
        return False


    if any(str(c) not in allowed_qids for c in citations):
        return False

    if answer != NOT_FOUND:
        if not citations:
            return False
        if not evidence:
            return False
        normalized_context = _normalize_ws(context)
        for ev in evidence:
            if not isinstance(ev, dict):
                return False
            qid = str(ev.get("question_id", ""))
            quote = ev.get("quote", "")
            if qid not in allowed_qids:
                return False
            if not isinstance(quote, str) or not quote.strip():
                return False

            if STRICT_EVIDENCE_QUOTES:
                if quote not in context and _normalize_ws(quote) not in normalized_context:
                    return False

    return True


def answer_with_context(query: str, retrieved):
    """Generate an answer and reject output that cannot be grounded in context."""
    if not retrieved:
        return NOT_FOUND, []

    context_blocks = []
    citations = []
    for r in retrieved:
        qid = str(r["meta"]["question_id"])
        title = r["meta"]["title"]
        citations.append(qid)
        context_blocks.append(f"[Source question_id={qid} | title={title}]\n{r['chunk']}")

    context = "\n\n---\n\n".join(context_blocks)
    allowed_qids = set(citations)

    llm_kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL
    llm = ChatOpenAI(**llm_kwargs)

    user_prompt = f"""
QUESTION:
{query}

CONTEXT:
{context}
""".strip()

    raw = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]).content.strip()
    if os.getenv("RAG_DEBUG") == "1":
        print("\n[DEBUG] Raw model output:\n", raw)

    try:
        payload = _safe_json_load(raw)
    except Exception:
        return NOT_FOUND, []

    if not _validate_response(payload, context=context, allowed_qids=allowed_qids):
        return NOT_FOUND, []

    answer = payload.get("answer", NOT_FOUND)
    cites = sorted({str(c) for c in payload.get("citations", []) if str(c) in allowed_qids})

    if not answer:
        return NOT_FOUND, []

    return answer, cites


def main():
    """CLI loop for grounded Q&A."""
    index, metadata, chunks = load_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Type a question (or 'exit'):")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() == "exit":
            break

        retrieved = retrieve(query, index, embed_model, chunks, metadata, k=24)
        if not retrieved:

            retrieved = retrieve(
                query,
                index,
                embed_model,
                chunks,
                metadata,
                k=24,
                relative_margin=0.8,
                answers_only=False,
                min_chunk_chars=40,
            )

        print("\nRetrieved sources:")
        for r in retrieved:
            rtype = r["meta"].get("type", "unknown")
            chunk_id = r["meta"].get("chunk_id", "-")
            print(
                f"- #{r['rank']} question_id={r['meta']['question_id']} "
                f"| type={rtype} | chunk_id={chunk_id} | {r['meta']['title']} "
                f"(distance={r['distance']:.4f})"
            )

        answer, cites = answer_with_context(query, retrieved)

        print("\nAnswer (grounded):")
        print(answer)
        print("\nCitations:", cites)


if __name__ == "__main__":
    main()
