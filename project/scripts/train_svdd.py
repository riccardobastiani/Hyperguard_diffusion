from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import generate
from project.geometry import HyperbolicProjector, LorentzConfig, LorentzManifoldOps
from project.hooks import HookConfig
from project.monitoring import HiddenTrajectoryExtractor
from project.svdd import DeepSVDDConfig, HyperbolicDeepSVDD
from project.utils import configure_logging, set_seed


def _prompt_from_row(row: dict) -> str:
    value = row.get("refined_behavior")
    if isinstance(value, str) and value.strip():
        return value
    raise KeyError(
        "Expected dataset row to contain a non-empty 'refined_behavior' prompt column. "
        f"Available columns: {sorted(row)}"
    )


def _safe_filter(row: dict) -> bool:
    return row.get("resource_dataset") == "alpaca"


def _format_prompts(tokenizer, prompts: list[str]) -> list[str]:
    formatted = []
    for prompt in prompts:
        message = {"role": "user", "content": prompt}
        formatted.append(tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False))
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hyperbolic Deep SVDD on safe LLaDA hidden trajectories.")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", default="saralazza/llada-safety-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--selected-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--selected-layers", nargs="+", default=["23"])
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--hyperbolic-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    dataset = load_dataset(args.dataset, split=args.split)
    safe_rows = dataset.filter(_safe_filter)
    safe_rows = safe_rows.select(range(min(args.max_examples, len(safe_rows))))

    selected_layers = tuple(int(x) if str(x).isdigit() else x for x in args.selected_layers)
    extractor = HiddenTrajectoryExtractor(
        HookConfig(
            selected_layers=selected_layers,
            selected_steps=tuple(args.selected_steps),
            offload_to_cpu=args.cpu_offload,
            capture_dtype=torch.float32,
        )
    )
    extractor.attach(model)

    hidden_chunks: list[torch.Tensor] = []
    try:
        for start in range(0, len(safe_rows), args.batch_size):
            rows = [safe_rows[index] for index in range(start, min(start + args.batch_size, len(safe_rows)))]
            prompts = [_prompt_from_row(row) for row in rows]
            formatted = _format_prompts(tokenizer, prompts)
            encoded = tokenizer(formatted, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            generate(
                model,
                input_ids,
                attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
                monitor=extractor,
            )
            hidden_chunks.extend(record.hidden.cpu() for record in extractor.records)
    finally:
        extractor.remove()

    if not hidden_chunks:
        raise RuntimeError("No hidden states were captured. Check selected layers and selected steps.")
    hidden = torch.cat(hidden_chunks, dim=0)

    lorentz = LorentzManifoldOps(LorentzConfig())
    projector = HyperbolicProjector(args.hidden_dim, args.hyperbolic_dim, lorentz=lorentz, bias=False)
    detector = HyperbolicDeepSVDD(projector, DeepSVDDConfig())
    detector.to(device)
    detector.initialize_center([hidden], device=device)

    loader = DataLoader(TensorDataset(hidden), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=1e-5)
    detector.train()
    for epoch in range(args.epochs):
        total = 0.0
        for (batch_hidden,) in loader:
            batch_hidden = batch_hidden.to(device)
            output = detector(batch_hidden)
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(output.loss.item()) * batch_hidden.shape[0]
        print(f"epoch={epoch + 1} loss={total / len(loader.dataset):.6f} radius={float(detector.radius.item()):.6f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector.save_checkpoint(
        args.output,
        metadata={
            "selected_layers": list(selected_layers),
            "selected_steps": args.selected_steps,
            "dataset": args.dataset,
            "split": args.split,
            "safe_filter": "resource_dataset == 'alpaca'",
        },
    )


if __name__ == "__main__":
    main()
