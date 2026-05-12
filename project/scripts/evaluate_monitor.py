from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import generate
from project.evaluation import classification_metrics, latency_overhead, optimal_stopping_step
from project.hooks import HookConfig
from project.monitoring import RuntimeJailbreakMonitor, RuntimeMonitorConfig
from project.svdd import HyperbolicDeepSVDD
from project.utils import configure_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLaDA trajectory monitor on a JSONL prompt set.")
    parser.add_argument("--jsonl", required=True, help="Each row must contain prompt and label, where label 1=attack.")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--selected-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--selected-layers", nargs="+", default=["15", "23", "31"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _read_rows(path: str, max_examples: int | None) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = _read_rows(args.jsonl, args.max_examples)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    detector = HyperbolicDeepSVDD.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    selected_layers = tuple(int(x) if str(x).isdigit() else x for x in args.selected_layers)
    monitor = RuntimeJailbreakMonitor(
        detector,
        HookConfig(selected_layers=selected_layers, selected_steps=tuple(args.selected_steps)),
        RuntimeMonitorConfig(
            selected_steps=tuple(args.selected_steps),
            step_weights={step: 1.0 / len(args.selected_steps) for step in args.selected_steps},
            threshold=args.threshold,
        ),
        device=device,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    stopping_steps: list[int | None] = []
    baseline_time = 0.0
    monitored_time = 0.0
    outputs: list[dict] = []

    for row in rows:
        prompt = row["prompt"]
        label = int(row["label"])
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        encoded = tokenizer([formatted], add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        t0 = time.perf_counter()
        generate(model, input_ids, attention_mask, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
        baseline_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        out = generate(
            model,
            input_ids,
            attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            monitor=monitor,
        )
        monitored_time += time.perf_counter() - t0

        stopped = bool(monitor.stopped)
        stopping_step = next((event["global_step"] for event in monitor.events if event["should_stop"]), None)
        y_true.append(label)
        y_pred.append(1 if stopped else 0)
        stopping_steps.append(stopping_step)
        outputs.append(
            {
                "prompt": prompt,
                "label": label,
                "stopped": stopped,
                "stopping_step": stopping_step,
                "score": monitor.events[-1]["score"] if monitor.events else 0.0,
                "generation": tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0],
            }
        )

    metrics = classification_metrics(y_true, y_pred)
    report = {
        "metrics": metrics.__dict__,
        "latency_overhead": latency_overhead(baseline_seconds=baseline_time, monitored_seconds=monitored_time),
        "optimal_stopping_step": optimal_stopping_step(stopping_steps, y_true, max_step=args.steps),
        "examples": outputs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
