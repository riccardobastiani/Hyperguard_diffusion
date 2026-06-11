from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path("datasets")

DATASETS = {
    "advbench": DATA_ROOT / "advbench" / "llada_instruct_DIJA_v1.json",
    "harmbench": DATA_ROOT / "harmbench" / "llada_jailbreak_dataset_harmbench.json",
    "jailbreakbench": DATA_ROOT / "jailbreakbench" / "llada_instruct_DIJA_v1.json",
    "strongreject": DATA_ROOT / "strongreject" / "llada_jailbreak_dataset_strongreject.json",
    "alpaca": DATA_ROOT / "alpaca" / "llada_instruct_DIJA_v1.json",
}

HF_USERNAME = "saralazza"
HF_DATASET_NAME = "llada-safety-dataset"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD + NORMALIZE
# =========================================================

all_rows = []

for source_name, file_path in DATASETS.items():

    print(f"Loading {source_name}...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):

        row = {
            "id": f"{source_name}_{idx:06d}",
            "source_dataset": source_name,
            "behavior": item.get("Behavior", ""),
            "refined_behavior": item.get("Refined_behavior", ""),
            "response": item.get("Response", ""),
            "language": "en",
        }

        all_rows.append(row)

# =========================================================
# DATAFRAME
# =========================================================

print("Creating dataframe...")

df = pd.DataFrame(all_rows)

print(f"Total rows before dedup: {len(df)}")

# =========================================================
# CLEANING
# =========================================================

# Remove exact duplicates

df = df.drop_duplicates(
    subset=["behavior", "refined_behavior", "response"]
)

# Remove empty rows

df = df[
    (df["behavior"].str.len() > 0)
    & (df["response"].str.len() > 0)
]

print(f"Total rows after dedup: {len(df)}")

# =========================================================
# SPLIT BY BEHAVIOR (ANTI-LEAKAGE)
# =========================================================

print("Creating train/val/test split...")

unique_behaviors = df["behavior"].dropna().astype(str).unique().tolist()

train_behaviors, temp_behaviors = train_test_split(
    unique_behaviors,
    test_size=(TEST_SIZE + VAL_SIZE),
    random_state=RANDOM_SEED,
)

relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)

val_behaviors, test_behaviors = train_test_split(
    temp_behaviors,
    test_size=relative_test_size,
    random_state=RANDOM_SEED,
)

train_set = set(train_behaviors)
val_set = set(val_behaviors)
test_set = set(test_behaviors)


def assign_split(behavior):
    if behavior in train_set:
        return "train"
    elif behavior in val_set:
        return "validation"
    else:
        return "test"


# Assign split

df["split"] = df["behavior"].apply(assign_split)

# =========================================================
# SAVE PARQUET
# =========================================================

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

print("Saving parquet files...")

train_path = OUTPUT_DIR / "train.parquet"
val_path = OUTPUT_DIR / "validation.parquet"
test_path = OUTPUT_DIR / "test.parquet"

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# =========================================================
# HUGGING FACE DATASET OBJECT
# =========================================================

print("Creating Hugging Face dataset...")

hf_dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    }
)

# =========================================================
# CREATE HF REPO
# =========================================================

repo_id = f"{HF_USERNAME}/{HF_DATASET_NAME}"

print(f"Creating repo: {repo_id}")

create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
)

# =========================================================
# PUSH TO HUB
# =========================================================

print("Uploading to Hugging Face Hub...")

hf_dataset.push_to_hub(repo_id, commit_message="Updated dataset with new preprocessing and dedup")

print("Done!")
print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
