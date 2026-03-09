"""Build and persist the retrieval index used by the RAG CLI.

This module:
1. Loads StackOverflow questions and answers.
2. Creates question/answer text chunks with overlap.
3. Encodes chunks into embeddings.
4. Stores vectors in FAISS plus aligned metadata/chunk files.
"""

from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from dotenv import load_dotenv

load_dotenv()


DATA_DIR = Path("data/stackoverflow")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and join question/answer CSVs into one dataframe."""
    questions = pd.read_csv(DATA_DIR / "Questions.csv", encoding="latin1")
    answers = pd.read_csv(DATA_DIR / "Answers.csv", encoding="latin1")

    # relevant columns
    questions = questions[["Id", "Title", "Body"]]
    answers = answers[["ParentId", "Body"]]

    merged = questions.merge(
        answers,
        left_on="Id",
        right_on="ParentId",
        how="inner",
        suffixes=("_question", "_answer"),
    )
    return merged


def build_chunks(df, max_paragraphs=3, overlap=1):
    """Create retrieval chunks and aligned metadata entries.

    Each joined row yields:
    - one question chunk
    - multiple answer chunks using a paragraph sliding window
    """
    chunks = []
    metadata = []

    if overlap >= max_paragraphs:
        raise ValueError("overlap must be smaller than max_paragraphs")

    for _, row in df.iterrows():
        qid = int(row["Id"])
        title = row["Title"]

        # Question Chunk
        question_chunk = f"""
QUESTION:
{title}

{row['Body_question']}
""".strip()

        chunks.append(question_chunk)
        metadata.append(
            {"question_id": qid, "title": title, "type": "question"})

        # Answer Chunks paragraph-based with overlap
        answer_text = row["Body_answer"]
        paragraphs = [p.strip()
                      for p in answer_text.split("\n\n") if p.strip()]

        i = 0
        chunk_id = 0

        while i < len(paragraphs):
            window = paragraphs[i: i + max_paragraphs]

            chunk_text = f"""
ANSWER:
{title}

{"\n\n".join(window)}
""".strip()

            chunks.append(chunk_text)
            metadata.append(
                {
                    "question_id": qid,
                    "title": title,
                    "type": "answer",
                    "chunk_id": chunk_id,
                }
            )

            chunk_id += 1
            i += max_paragraphs - overlap

    return chunks, metadata


def main():
    """Build embeddings + FAISS index and save index artifacts to disk."""
    print("Loading data...")
    df = load_data().head(5000)  # start small to avoid overloading your machine

    print("Creating chunks...")
    chunks, metadata = build_chunks(df)

    print(
        "Loading embedding model...That can convert text into vectors (embeddings) for similarity search (RAG/FAISS)."
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Build FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    #
    print("Saving ... binary file metadata.pkl")
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Saving chunks....binary file chunks.pkl")
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Index built successfully")


if __name__ == "__main__":
    main()
