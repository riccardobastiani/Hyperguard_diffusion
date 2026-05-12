from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import generate
from project.hooks import HookConfig
from project.monitoring import RuntimeJailbreakMonitor, RuntimeMonitorConfig
from project.svdd import HyperbolicDeepSVDD
from project.utils import configure_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaDA generation with real-time trajectory monitoring.")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--selected-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--selected-layers", nargs="+", default=["15", "23", "31"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

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

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer([formatted], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    out = generate(
        model,
        input_ids,
        attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        monitor=monitor,
    )
    print(tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0])
    print({"stopped": monitor.stopped, "score": monitor.events[-1]["score"] if monitor.events else 0.0})


if __name__ == "__main__":
    main()
