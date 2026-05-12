from pathlib import Path
from datasets import load_dataset
from collections import Counter


# =========================
# Configuration
# =========================

DATASET_NAME = "saralazza/llada-safety-dataset"
SAFE_SOURCE_NAME = "alpaca"
OUTPUT_DIR = Path("dataset")


# =========================
# Create output directory
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Output directory: {OUTPUT_DIR.resolve()}")


# =========================
# Download dataset
# =========================

print(f"[INFO] Downloading dataset: {DATASET_NAME}")

dataset = load_dataset(DATASET_NAME)

print("[INFO] Dataset downloaded successfully.\n")


# =========================
# Print dataset statistics
# =========================

print("========== DATASET STATISTICS ==========\n")

global_total = 0

for split_name, split_dataset in dataset.items():

    print(f"========== SPLIT: {split_name.upper()} ==========")

    num_samples = len(split_dataset)
    global_total += num_samples

    print(f"Total samples: {num_samples}")

    # Count samples by source_dataset
    source_counter = Counter(split_dataset["source_dataset"])

    print("\nSamples by source_dataset:\n")

    for source_name, count in sorted(
        source_counter.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / num_samples) * 100

        print(
            f"  - {source_name:<20} "
            f"{count:>5} samples "
            f"({percentage:.2f}%)"
        )

    safe_count = sum(
        count
        for source_name, count in source_counter.items()
        if (source_name or "").strip().lower() == SAFE_SOURCE_NAME
    )
    unsafe_count = num_samples - safe_count

    print("\nSafety split counts:")
    print(f"  - safe (source_dataset={SAFE_SOURCE_NAME}): {safe_count}")
    print(f"  - unsafe (all others): {unsafe_count}\n")


print("========================================")
print(f"Global total samples: {global_total}")
print("========================================\n")


# =========================
# Save parquet files
# =========================

print("[INFO] Saving parquet files...\n")

for split_name, split_dataset in dataset.items():

    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    split_dataset.to_parquet(str(output_path))

    print(f"[SAVED] {split_name} -> {output_path}")


print("\n[DONE] Dataset export completed successfully.")