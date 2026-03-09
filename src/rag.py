"""RAG query runtime with retrieval diagnostics and grounded answer validation.

This module:
1. Loads the saved FAISS index + metadata/chunks.
2. Retrieves relevant chunks for a user query.
3. Calls an LLM with strict JSON + citation constraints.
4. Validates grounding before returning an answer.
"""

from pathlib import Path
import json
import os
import pickle
from collections import defaultdict
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
RETRIEVAL_DEFAULTS = {
    "k": 24,
    "max_distance": 0.3,
    "relative_margin": 0.25,
    "max_chunks_per_qid": 4,
    "answers_only": True,
    "min_chunk_chars": 80,
}

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
      "quote":  quote copied verbatim from CONTEXT"
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
    k: int = RETRIEVAL_DEFAULTS["k"],
    max_distance: float = RETRIEVAL_DEFAULTS["max_distance"],
    relative_margin: float = RETRIEVAL_DEFAULTS["relative_margin"],
    max_chunks_per_qid: int = RETRIEVAL_DEFAULTS["max_chunks_per_qid"],
    answers_only: bool = RETRIEVAL_DEFAULTS["answers_only"],
    min_chunk_chars: int = RETRIEVAL_DEFAULTS["min_chunk_chars"],
):
    """Retrieve chunks and apply strict distance gating to reduce irrelevant context."""
    debug_retrieve = os.getenv("RAG_RETRIEVE_DEBUG", "0") == "1"
    debug_verbose = os.getenv("RAG_RETRIEVE_DEBUG_VERBOSE", "0") == "1"

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, k)

    raw_total = 0
    rejected_no_index = 0
    rejected_max_distance = 0
    valid_pairs = []
    for rank, idx in enumerate(indices[0]):
        raw_total += 1
        if idx == -1:
            rejected_no_index += 1
            continue
        dist = float(distances[0][rank])
        if dist <= max_distance:
            valid_pairs.append((rank, idx, dist))
        else:
            rejected_max_distance += 1

    if not valid_pairs:
        if debug_retrieve:
            print("\n[RETRIEVE DEBUG] query:", query)
            print(
                f"[RETRIEVE DEBUG] raw_topk={raw_total} | pass_max_distance=0 | final=0")
            print(
                "[RETRIEVE DEBUG] rejected:"
                f" no_index={rejected_no_index}"
                f" max_distance={rejected_max_distance}"
                " relative_margin=0 answers_only=0 min_chunk_chars=0 per_qid=0"
            )
            print(
                "[RETRIEVE DEBUG] params:"
                f" k={k} max_distance={max_distance}"
                f" relative_margin={relative_margin}"
                f" max_chunks_per_qid={max_chunks_per_qid}"
                f" answers_only={answers_only}"
                f" min_chunk_chars={min_chunk_chars}"
            )
        return []

    best_dist = min(d for _, _, d in valid_pairs)

    per_qid_counts = {}
    results = []
    rejected_margin = 0
    rejected_type = 0
    rejected_len = 0
    rejected_per_qid = 0
    inspected_rows = []
    for rank, idx, dist in valid_pairs:
        meta = metadata[idx]
        qid = str(meta["question_id"])
        mtype = str(meta.get("type", ""))
        chunk_text = str(chunks[idx])
        reason = "accepted"

        if dist > best_dist + relative_margin:
            rejected_margin += 1
            reason = "relative_margin"
            if debug_verbose:
                inspected_rows.append((rank + 1, qid, mtype, dist, reason))
            continue

        if answers_only and mtype != "answer":
            rejected_type += 1
            reason = "answers_only"
            if debug_verbose:
                inspected_rows.append((rank + 1, qid, mtype, dist, reason))
            continue

        if len(chunk_text.strip()) < min_chunk_chars:
            rejected_len += 1
            reason = "min_chunk_chars"
            if debug_verbose:
                inspected_rows.append((rank + 1, qid, mtype, dist, reason))
            continue

        count = per_qid_counts.get(qid, 0)
        if count >= max_chunks_per_qid:
            rejected_per_qid += 1
            reason = "max_chunks_per_qid"
            if debug_verbose:
                inspected_rows.append((rank + 1, qid, mtype, dist, reason))
            continue
        per_qid_counts[qid] = count + 1

        results.append(
            {
                "rank": rank + 1,
                "distance": dist,
                "chunk": chunk_text,
                "meta": meta,
            }
        )
        if debug_verbose:
            inspected_rows.append((rank + 1, qid, mtype, dist, reason))

    results.sort(key=lambda r: (r["distance"]))
    if debug_retrieve:
        answer_count = sum(
            1 for r in results if str(r["meta"].get("type", "")) == "answer"
        )
        question_count = sum(
            1 for r in results if str(r["meta"].get("type", "")) == "question"
        )
        print("\n[RETRIEVE DEBUG] query:", query)
        print(
            f"[RETRIEVE DEBUG] raw_topk={raw_total}"
            f" | pass_max_distance={len(valid_pairs)}"
            f" | final={len(results)}"
            f" (answer={answer_count}, question={question_count})"
        )
        print(
            "[RETRIEVE DEBUG] rejected:"
            f" no_index={rejected_no_index}"
            f" max_distance={rejected_max_distance}"
            f" relative_margin={rejected_margin}"
            f" answers_only={rejected_type}"
            f" min_chunk_chars={rejected_len}"
            f" per_qid={rejected_per_qid}"
        )
        print(
            "[RETRIEVE DEBUG] params:"
            f" k={k} max_distance={max_distance}"
            f" best_dist={best_dist:.4f}"
            f" relative_margin={relative_margin}"
            f" max_chunks_per_qid={max_chunks_per_qid}"
            f" answers_only={answers_only}"
            f" min_chunk_chars={min_chunk_chars}"
        )
        if results:
            preview = ", ".join(
                [
                    (
                        f"#{r['rank']} qid={r['meta'].get('question_id')} "
                        f"type={r['meta'].get('type', 'unknown')} d={r['distance']:.4f}"
                    )
                    for r in results[:8]
                ]
            )
            print(f"[RETRIEVE DEBUG] accepted_preview: {preview}")
        if debug_verbose and inspected_rows:
            print("[RETRIEVE DEBUG] candidate_decisions:")
            for rank, qid, mtype, dist, reason in inspected_rows:
                print(
                    f"  - rank={rank} qid={qid} type={mtype} dist={dist:.4f} -> {reason}"
                )
    return results


def _safe_json_load(text: str):
    """Parse model output as JSON, with fallback for fenced/wrapped content."""
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
    """Normalize whitespace to make quote checks more tolerant."""
    return " ".join(text.split())


def _validate_response(payload, context: str, allowed_qids: set[str]):
    """Validate model output shape and grounding constraints."""
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
                if (
                    quote not in context
                    and _normalize_ws(quote) not in normalized_context
                ):
                    return False

    return True


def _deduplicate_by_question_keep_chunks(retrieved, max_chunks_per_qid=3):
    """
    Keep the best N chunks per question_id while preserving ranking order.
    """
    grouped = defaultdict(list)

    for r in retrieved:
        qid = r["meta"]["question_id"]
        grouped[qid].append(r)

    result = []

    for qid, items in grouped.items():

        items_sorted = sorted(items, key=lambda x: x["distance"])
        result.extend(items_sorted[:max_chunks_per_qid])

    result = sorted(result, key=lambda x: x["distance"])

    return result


def answer_with_context(query: str, retrieved):
    """Generate an answer and reject output that cannot be grounded in context."""
    if not retrieved:
        return NOT_FOUND, []

    grouped = defaultdict(list)
    for r in retrieved:
        qid = str(r["meta"]["question_id"])
        grouped[qid].append(r)

    selected = []
    for qid in grouped:
        selected.extend(grouped[qid][:3])

    selected.sort(key=lambda r: r["distance"])

    context_blocks = []
    citations = set()
    for r in selected:
        qid = str(r["meta"]["question_id"])
        title = r["meta"]["title"]
        citations.add(qid)
        context_blocks.append(
            f"[Source question_id={qid} | title={title}]\n{r['chunk']}"
        )

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

    raw = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    ).content.strip()
    if os.getenv("RAG_DEBUG") == "1":
        print("\n[DEBUG] Raw model output:\n", raw)

    try:
        payload = _safe_json_load(raw)
    except Exception:
        return NOT_FOUND, []

    if not _validate_response(
            payload,
            context=context,
            allowed_qids=allowed_qids):
        return NOT_FOUND, []

    answer = payload.get("answer", NOT_FOUND)
    cites = sorted({str(c) for c in payload.get(
        "citations", []) if str(c) in allowed_qids})

    if not answer:
        return NOT_FOUND, []

    return answer, cites


def main():
    """CLI loop for grounded Q&A."""
    index, metadata, chunks = load_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    retrieval_kwargs = dict(RETRIEVAL_DEFAULTS)

    print("Type a question (or 'exit'):")
    while True:
        query = input("\n> ").strip()
        if not query or query.lower() == "exit":
            break

        retrieved = retrieve(
            query, index, embed_model, chunks, metadata, **retrieval_kwargs
        )
        retrieved = _deduplicate_by_question_keep_chunks(retrieved)

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
