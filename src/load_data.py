from pathlib import Path
import pandas as pd

def load_questions(csv_path: str | Path, n: int = 20) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="latin1")
    
    required = {"Id", "Title", "Body"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Questions.csv: {missing}")
    
    df = df.dropna(subset=["Id", "Title", "Body"]).head(n).copy()
    df["Id"] = df["Id"].astype(int)
    return df
