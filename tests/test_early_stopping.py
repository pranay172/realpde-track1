import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess

import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
sys.path.insert(0,str(REPO/"scripts"))
from realpde_t1.early_stopping import PlateauMonitor
from backbone_plateau import finalize_early, trainer_running, stop_owned_child


class EarlyStoppingTests(unittest.TestCase):
    def test_owned_child_does_not_inspect_transient_cmdline(self):
        child=Mock()
        child.poll.return_value=None
        with patch("backbone_plateau.check_process",side_effect=AssertionError("must not inspect proc")):
            self.assertTrue(trainer_running(child,123,Path("config")))
            child.poll.return_value=1
            self.assertFalse(trainer_running(child,123,Path("config")))

    def test_attached_process_still_validates_identity(self):
        with patch("backbone_plateau.check_process",return_value=True) as check:
            self.assertTrue(trainer_running(None,123,Path("config")))
            check.assert_called_once_with(123,Path("config"))

    def test_cleanup_is_bounded_and_never_kills_adopted_trainer(self):
        stop_owned_child(None)
        child=Mock()
        child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired("trainer",30),0]
        stop_owned_child(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_args_list[-1].kwargs,{"timeout":10})

    def test_warmup_and_plateau(self):
        m=PlateauMonitor()
        self.assertFalse(m.observe([1.]*100,100)["ready"])
        self.assertTrue(m.observe([1.]*200,200)["improved"])
        self.assertFalse(m.observe([1.]*800,800)["stop"])
        self.assertTrue(m.observe([1.]*1000,1000)["stop"])

    def test_material_improvement_resets_patience(self):
        m=PlateauMonitor()
        m.observe([1.]*200,200)
        self.assertFalse(m.observe([.995]*200,700)["improved"])
        self.assertTrue(m.observe([.98]*200,800)["improved"])
        self.assertFalse(m.observe([.98]*200,1300)["stop"])
        self.assertTrue(m.observe([.98]*200,1400)["stop"])

    def test_smoothing_and_finiteness(self):
        m=PlateauMonitor()
        self.assertAlmostEqual(m.observe([0.,2.]*100,200)["mean_loss"],1.)
        with self.assertRaises(ValueError):
            m.observe([float("nan")]*200,400)

    def test_finalize_preserves_best_and_labels_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            split=folder/"split.json"
            split.write_text(json.dumps({"training_files":["train.h5"]}))
            cfg={"experiment":"TEST","seed":1,"split_config":str(split),"training":{"num_updates":3000}}
            config=folder/"config.json"
            config.write_text(json.dumps(cfg))
            digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            torch.save({"update":400,"model_state_dict":{"weight":torch.tensor([2.])},
                        "normalizer":{},"config_sha256":digest(config),"split_config_sha256":digest(split)},
                       folder/"best_plateau_recovery.pth")
            (folder/"early_stopping_policy.json").write_text(json.dumps({"validation_used":False}))
            finalize_early(folder,config,cfg,{"update":1000})
            checkpoint=torch.load(folder/"final.pth",weights_only=False)
            self.assertEqual(checkpoint["status"],"early_stopped_final")
            self.assertEqual(checkpoint["training"]["updates"],400)
            self.assertEqual(checkpoint["model_state_dict"]["weight"].item(),2.)
            self.assertEqual(json.loads((folder/"train_report.json").read_text())["status"],"early_stopped")
            with self.assertRaises(FileExistsError):
                finalize_early(folder,config,cfg,{"update":1000})
