import unittest

import torch
from torch import nn

from project.hooks import HiddenStateCapture, HookConfig


class DummyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DummyModel(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock(dim) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HiddenStateHookTest(unittest.TestCase):
    def test_capture_pooled_layer_output(self) -> None:
        model = DummyModel()
        capture = HiddenStateCapture(model, HookConfig(selected_layers=(1,), selected_steps=(5,)))
        capture.attach()
        try:
            attention_mask = torch.ones(2, 4)
            capture.start_step(global_step=5, block_index=0, local_step=4, batch_size=2, attention_mask=attention_mask)
            model(torch.randn(2, 4, 8))
            records = capture.finish_step()
        finally:
            capture.remove()

        self.assertEqual(len(records), 1)
        self.assertEqual(tuple(records[0].hidden.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
