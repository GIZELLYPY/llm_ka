"""run pipeline + save JSONL (json lines)
example:
{"id": 1, "title": "A"}
{"id": 2, "title": "B"}
{"id": 3, "title": "C"}

easy to stream/process line by line
good for large datasets
append-friendly (just add new lines)
"""


from pathlib import Path
import json
from tqdm import tqdm

from .load_data import load_questions
from .extractor import extract_one
from .retry import retry_once

DATA_DIR = Path("data/stackoverflow")
OUT_PATH = Path("data/extracted_questions.jsonl")
FAILED_PATH =  Path("data/failed_extractions.jsonl") 

def main():
    questions_path = DATA_DIR / "Questions.csv"
    df = load_questions(questions_path, n=20)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
    OUT_PATH.open("w", encoding="utf-8") as ok_f,
    FAILED_PATH.open("w", encoding="utf-8") as fail_f
    ):
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting"):
            qid = int(getattr(row, "Id"))
            title = str(getattr(row, "Title"))
            body = str(getattr(row, "Body"))
            
            def _call():
                return extract_one(qid, title, body) # (LLM + validation)
            
            try:
                extracted = retry_once(_call, sleep_s=1.0)
                ok_f.write(json.dumps(extracted.model_dump(), ensure_ascii=False) + "\n")
                
            
            except Exception as e:
                fail_record = {
                    "question_id": qid,
                    "title": title,
                    "error": str(e),
                    "body_preview": body[:500],
                }
                fail_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
                

        print(f" Saved successes: {OUT_PATH.resolve()}")
        print(f" Saved failures:  {FAILED_PATH.resolve()}")
    


if __name__ == "__main__":
    main()