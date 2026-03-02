"""
What metadata.pkl Looks Like:
[
  {
    "question_id": 1040,
    "title": "How do I delete a file which is locked by another process in C#?",
    "type": "question"
  },
  {
    "question_id": 1040,
    "title": "How do I delete a file which is locked by another process in C#?",
    "type": "answer",
    "chunk_id": 0
  },
  {
    "question_id": 1040,
    "title": "How do I delete a file which is locked by another process in C#?",
    "type": "answer",
    "chunk_id": 1
  },
  {
    "question_id": 205,
    "title": "SQL GROUP BY with HAVING",
    "type": "question"
  }
]

------------------------------------------------------------
------------------------------------------------------------
faiss.index Stores

FAISS does not know anything about:

question_id

titles

types

chunk_id

It only stores vectors.

vector 0 → chunks[0] → metadata[0]
vector 1 → chunks[1] → metadata[1]
vector 2 → chunks[2] → metadata[2]
vector 3 → chunks[3] → metadata[3]

------------------------------------------------------------
------------------------------------------------------------

What Happens During Search? 

"how do I delete a locked file in C#?"

1: The query is embedded into a vector.
2: FAISS computes L2 distance between query vector and all stored vectors
3: FAISS returns closest vector indices: [2, 1, 0]
4: the code map back: metadata[2], metadata[1], metadata[0]


Which might correspond to: answer chunk 1, answer chunk 0, question chunk

------------------------------------------------------------
------------------------------------------------------------

FAISS answers: “Which fragments are semantically closest?”

metadata.pkl answers: “What does this fragment represent?”

FAISS gives proximity.
Metadata gives meaning.
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
        metadata.append({
            "question_id": qid,
            "title": title,
            "type": "question"
        })


        # Answer Chunks paragraph-based with overlap
        answer_text = row["Body_answer"]
        paragraphs = [p.strip() for p in answer_text.split("\n\n") if p.strip()]

        i = 0
        chunk_id = 0

        while i < len(paragraphs):
            window = paragraphs[i:i + max_paragraphs]

            chunk_text = f"""
ANSWER:
{title}

{"\n\n".join(window)}
""".strip()

            chunks.append(chunk_text)
            metadata.append({
                "question_id": qid,
                "title": title,
                "type": "answer",
                "chunk_id": chunk_id
            })

            chunk_id += 1
            i += max_paragraphs - overlap  # sliding window

    return chunks, metadata


def main():
    print("Loading data...")
    df = load_data().head(500)  # start small

    print("Creating chunks...")
    chunks, metadata = build_chunks(df)

    print(
        "Loading embedding model...That can convert text into vectors (embeddings) for similarity search (RAG/FAISS)."
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Converts each chunk into a numeric vector.
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Uses faiss.IndexFlatL2 (L2 distance).
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
