import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset


LOGGER = logging.getLogger(__name__)


def _resolve_split(dataset: DatasetDict, preferred_split: str) -> Dataset:
    """Return the preferred split, or the first available split."""
    if preferred_split in dataset:
        return dataset[preferred_split]

    split_name = next(iter(dataset.keys()))
    LOGGER.warning("Split '%s' not found. Using '%s' instead.", preferred_split, split_name)
    return dataset[split_name]


def _safe_alpaca_prompt(row) -> str:
    """Build a benign prompt from Alpaca instruction and optional input."""
    instruction = (row.get("instruction") or "").strip()
    input_text = (row.get("input") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def _collect_prompts(dataset: Dataset, text_builder, sample_count: int) -> List[str]:
    """Collect non-empty prompts from a shuffled Hugging Face dataset."""
    prompts = []
    for row in dataset:
        prompt = text_builder(row).strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) == sample_count:
            break

    if len(prompts) < sample_count:
        raise ValueError(f"Requested {sample_count} prompts, but only found {len(prompts)} usable rows.")
    return prompts


def _load_local_prompts(data_path: Path, field_name: str, sample_count: int, seed: int) -> List[str]:
    """Load prompts from a local JSON/CSV file and return a deterministic sample."""
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        dataset = Dataset.from_json(str(data_path))
    elif suffix == ".csv":
        dataset = Dataset.from_csv(str(data_path))
    else:
        raise ValueError(f"Unsupported local dataset format: {data_path}. Use .json or .csv.")

    dataset = dataset.shuffle(seed=seed)
    return _collect_prompts(dataset, lambda row: (row.get(field_name) or ""), sample_count)


def load_balanced_prompt_dataset(
    samples_per_class: int = 50,
    seed: int = 42,
    safe_dataset_name: str = "tatsu-lab/alpaca",
    safe_local_path: str | None = None,
    safe_local_field: str = "Refined_behavior",
    unsafe_dataset_name: str = "saralazza/llada-safety-dataset",
    safe_split: str = "train",
    unsafe_split: str = "train",
) -> Tuple[List[str], List[int]]:
    """
    Load balanced safe and unsafe prompts.

    Safe prompts come from a local JSON/CSV field when provided; otherwise they
    come from Alpaca using instruction plus input. Unsafe prompts come from the
    LLaDA safety dataset using refined_behavior. Labels are safe=0 and unsafe=1.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")

    if safe_local_path:
        safe_path = Path(safe_local_path)
        LOGGER.info("Loading local safe dataset: %s (%s)", safe_path, safe_local_field)
        safe_prompts = _load_local_prompts(safe_path, safe_local_field, samples_per_class, seed)
    else:
        LOGGER.info("Loading safe dataset: %s", safe_dataset_name)
        safe_dataset = _resolve_split(load_dataset(safe_dataset_name), safe_split)
        safe_dataset = safe_dataset.shuffle(seed=seed)
        safe_prompts = _collect_prompts(safe_dataset, _safe_alpaca_prompt, samples_per_class)

    LOGGER.info("Loading unsafe dataset: %s", unsafe_dataset_name)
    unsafe_dataset = _resolve_split(load_dataset(unsafe_dataset_name), unsafe_split)
    unsafe_dataset = unsafe_dataset.shuffle(seed=seed)
    unsafe_prompts = _collect_prompts(
        unsafe_dataset,
        lambda row: (row.get("refined_behavior") or ""),
        samples_per_class,
    )

    examples = [(prompt, 0) for prompt in safe_prompts]
    examples.extend((prompt, 1) for prompt in unsafe_prompts)

    rng = random.Random(seed)
    rng.shuffle(examples)

    prompts, labels = zip(*examples)
    LOGGER.info("Prepared %d prompts (%d safe, %d unsafe).", len(prompts), samples_per_class, samples_per_class)
    return list(prompts), list(labels)


def validate_labels(labels: Sequence[int]) -> None:
    """Ensure labels contain both expected classes."""
    label_set = set(labels)
    if label_set != {0, 1}:
        raise ValueError(f"Expected labels {{0, 1}}, got {sorted(label_set)}.")
