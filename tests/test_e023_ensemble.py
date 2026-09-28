import unittest
import torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from realpde_t1.ensemble import (
    SharedBackboneResidualEnsemble,
    assert_bundle_matches_sources,
    build_ensemble_bundle,
    build_residual_adapter_from_config,
)
from load_baseline import build_model


class TestE023Ensemble(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.backbone = build_model("cno", device="cpu")
        self.cfg1 = {
            "name": "residual_conv3d",
            "in_channels": 6,
            "out_channels": 3,
            "hidden_channels": 16,
            "n_blocks": 2,
            "kernel_size": 3,
            "zero_init_last": True,
        }
        self.cfg2 = {
            "name": "heteroskedastic_residual_conv3d",
            "in_channels": 6,
            "out_channels": 3,
            "uncertainty_channels": 2,
            "hidden_channels": 32,
            "n_blocks": 2,
            "kernel_size": 3,
            "zero_init_last": True,
            "initial_log_var": -4.0,
        }
        self.adapter1 = build_residual_adapter_from_config(self.cfg1)
        self.adapter2 = build_residual_adapter_from_config(self.cfg2)
        self.ensemble = SharedBackboneResidualEnsemble(
            self.backbone, [self.adapter1, self.adapter2], [self.cfg1, self.cfg2]
        )

    def test_ensemble_forward_shape(self):
        x = torch.randn(2, 20, 32, 64, 3)
        with torch.no_grad():
            pred, var = self.ensemble(x)
        self.assertEqual(pred.shape, (2, 20, 32, 64, 3))
        self.assertEqual(var.shape, (2, 20, 32, 64, 2))
        self.assertTrue(torch.all(var >= 0.0))

    def test_zero_init_identity(self):
        """When adapters are zero-initialized, ensemble output equals base backbone output."""
        x = torch.randn(1, 20, 32, 64, 3)
        with torch.no_grad():
            base = self.backbone(x)
            pred, _ = self.ensemble(x)
        self.assertTrue(torch.allclose(pred, base, atol=1e-6))

    def test_bundle_builder_requires_identical_backbones_and_preserves_adapters(self) -> None:
        backbone = {"w": torch.tensor([1.0, 2.0])}
        cfg = {"name": "residual_conv3d", "hidden_channels": 16}
        first = {
            "adapter_config": cfg,
            "model_state_dict": {
                "backbone.w": backbone["w"].clone(),
                "adapter.out.weight": torch.tensor([0.25]),
            },
        }
        second = {
            "adapter_config": dict(cfg),
            "model_state_dict": {
                "backbone.w": backbone["w"].clone(),
                "adapter.out.weight": torch.tensor([0.50]),
            },
        }
        bundle = build_ensemble_bundle([first, second], source_sha256=["a", "b"])
        self.assertEqual(bundle["model_kind"], "shared_backbone_residual_ensemble")
        self.assertTrue(torch.equal(bundle["backbone_state_dict"]["w"], backbone["w"]))
        self.assertEqual(len(bundle["adapters"]), 2)
        self.assertTrue(torch.equal(bundle["adapters"][1]["state_dict"]["out.weight"], torch.tensor([0.50])))
        assert_bundle_matches_sources(bundle, [first, second])
        second["model_state_dict"]["backbone.w"] = torch.tensor([9.0, 9.0])
        with self.assertRaisesRegex(AssertionError, "backbone tensor mismatch"):
            build_ensemble_bundle([first, second])

    def test_frozen_e024_bundle_matches_source_checkpoints(self) -> None:
        sources = [
            REPO / "artifacts" / "e019_residual_adapter_e005" / "final.pth",
            REPO / "artifacts" / "e022_heteroskedastic_residual" / "final.pth",
            REPO / "artifacts" / "e023_adapter3" / "final.pth",
        ]
        bundle_path = REPO / "artifacts" / "e023_ensemble" / "ensemble_bundle.pth"
        if not bundle_path.is_file() or any(not path.is_file() for path in sources):
            self.skipTest("local E019/E022/E023 checkpoints or ensemble bundle are not present")
        checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in sources]
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        assert_bundle_matches_sources(bundle, checkpoints)


if __name__ == "__main__":
    unittest.main()
