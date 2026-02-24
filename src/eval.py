from pathlib import Path
import json
import pandas as pd

EXTRACTED_PATH = Path("data/extracted_questions.jsonl")
FAILED_PATH = Path("data/failed_extractions.jsonl")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    df = read_jsonl(EXTRACTED_PATH)
    df_fail = read_jsonl(FAILED_PATH)

    if df.empty:
        print("No extracted records found. Run  python -m src.run_extract")
    
    if "primary_technology" in df.columns:
        primary_null_rate = df["primary_technology"].isna().mean()
        primary_empty_rate = (df["primary_technology"].fillna("").str.strip() == "").mean()
        primary_effective_null = max(primary_null_rate, primary_empty_rate)
    else:
        primary_effective_null = None


    problem_dist = None
    if "problem_type" in df.columns:
        problem_dist = df["problem_type"].fillna("null").value_counts(dropna=False)


    avg_conf = df["confidence"].mean() if "confidence" in df.columns else None


    unknown_diff_pct = None
    if "difficulty" in df.columns:
        unknown_diff_pct = (df["difficulty"].fillna("unknown") == "unknown").mean()

    print("\n EVALUATION RESULTS")
    print(f"- Records extracted: {len(df)}")
    print(f"- Records failed:    {len(df_fail)}")

    if primary_effective_null is not None:
        print(f"- primary_technology null/empty rate: {primary_effective_null:.2%}")
    else:
        print("- primary_technology null/empty rate: (column missing)")

    if avg_conf is not None:
        print(f"- average confidence: {avg_conf:.3f}")
    else:
        print("- average confidence: (column missing)")

    if unknown_diff_pct is not None:
        print(f"- difficulty == 'unknown': {unknown_diff_pct:.2%}")
    else:
        print("- difficulty == 'unknown': (column missing)")

    if problem_dist is not None:
        print("\n- problem_type distribution:")
        for k, v in problem_dist.items():
            print(f"  • {k}: {v}")
    else:
        print("\n- problem_type distribution: (column missing)")

if __name__ == "__main__":
    main()