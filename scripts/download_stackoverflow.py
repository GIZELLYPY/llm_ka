import kagglehub
from pathlib import Path


DATASET = "stackoverflow/stacksample"


path = kagglehub.dataset_download(DATASET)


dest = Path("data/stackoverflow")
dest.mkdir(parents=True, exist_ok=True)

print(f"Dataset downloaded to: {path}")

