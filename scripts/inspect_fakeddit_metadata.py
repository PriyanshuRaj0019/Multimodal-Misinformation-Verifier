
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/fakeddit/metadata")

FILES = [
    "multimodal_train.tsv",
    "multimodal_validate.tsv",
    "multimodal_test_public.tsv",
]

for filename in FILES:
    file_path = DATA_DIR / filename

    df = pd.read_csv(file_path, sep="\t", nrows=5)

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}")
    print(f"COLUMNS: {df.columns.tolist()}")
    print(f"FIRST 5 ROWS:")
    print(df.head().to_string(index=False))
