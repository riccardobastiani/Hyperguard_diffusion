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
    refined = (row.get("refined_behavior") or "").strip()
    if refined:
        return refined

    behavior = (row.get("behavior") or "").strip()
    if behavior:
        return behavior

    instruction = (row.get("instruction") or "").strip()
    input_text = (row.get("input") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def _unsafe_prompt(row) -> str:
    """Build an unsafe prompt with fallbacks for heterogeneous datasets."""
    refined = (row.get("refined_behavior") or "").strip()
    if refined:
        return refined

    prompt = (row.get("prompt") or "").strip()
    if prompt:
        return prompt

    behavior = (row.get("behavior") or "").strip()
    if behavior:
        return behavior

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
    safety_dataset_name: str = "saralazza/llada-safety-dataset",
    safe_local_path: str | None = None,
    safe_local_field: str = "Refined_behavior",
    safe_split: str = "train",
    unsafe_split: str = "train",
    safe_source_name: str = "alpaca",
) -> Tuple[List[str], List[int]]:
    """
    Load balanced safe and unsafe prompts.

    Safe prompts come from a local JSON/CSV field when provided; otherwise they
    come from the LLaDA safety dataset where source_dataset == safe_source_name
    using instruction plus input. Unsafe prompts come from the same dataset
    using rows where source_dataset != safe_source_name. Labels are safe=0 and
    unsafe=1.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")

    dataset_dict = None
    if safe_local_path:
        safe_path = Path(safe_local_path)
        LOGGER.info("Loading local safe dataset: %s (%s)", safe_path, safe_local_field)
        safe_prompts = _load_local_prompts(safe_path, safe_local_field, samples_per_class, seed)
    else:
        LOGGER.info("Loading safety dataset: %s", safety_dataset_name)
        dataset_dict = load_dataset(safety_dataset_name)

        safe_dataset = _resolve_split(dataset_dict, safe_split)
        safe_dataset = safe_dataset.filter(
            lambda row: (row.get("source_dataset") or "").strip().lower() == safe_source_name.lower()
        ).shuffle(seed=seed)
        safe_prompts = _collect_prompts(safe_dataset, _safe_alpaca_prompt, samples_per_class)

    LOGGER.info("Loading unsafe samples from: %s", safety_dataset_name)
    if dataset_dict is None:
        dataset_dict = load_dataset(safety_dataset_name)
    unsafe_dataset = _resolve_split(dataset_dict, unsafe_split)
    unsafe_dataset = unsafe_dataset.filter(
        lambda row: (row.get("source_dataset") or "").strip().lower() != safe_source_name.lower()
    ).shuffle(seed=seed)
    unsafe_prompts = _collect_prompts(unsafe_dataset, _unsafe_prompt, samples_per_class)

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
