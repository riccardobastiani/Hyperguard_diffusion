import unittest

import torch

from project.geometry import HyperbolicProjector, LorentzConfig, LorentzManifoldOps


class LorentzGeometryTest(unittest.TestCase):
    def test_expmap_outputs_valid_lorentz_points(self) -> None:
        ops = LorentzManifoldOps(LorentzConfig())
        tangent = torch.randn(4, 8)
        point = ops.expmap0(tangent)
        result = ops.validate(point)
        self.assertTrue(result.finite)
        self.assertLess(result.max_constraint_error, 1e-3)
        self.assertGreater(result.min_time, 0.0)

    def test_projector_shape(self) -> None:
        projector = HyperbolicProjector(hidden_dim=16, hyperbolic_dim=8)
        hidden = torch.randn(3, 16)
        point = projector(hidden)
        self.assertEqual(tuple(point.shape), (3, 9))


if __name__ == "__main__":
    unittest.main()
