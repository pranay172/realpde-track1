import copy
import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"scripts"))
from e042_protocol_preflight import validate_roles


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((REPO/"configs/e042_locked_protocol.json").read_text())
        self.split=json.loads((REPO/"configs/e042_point_split.json").read_text())
        source=json.loads((REPO/"configs/e002_split.json").read_text())
        self.universe=source["training_files"]+source["validation_files"]

    def test_partition(self):
        self.assertEqual(validate_roles(self.protocol,self.split,self.universe),
                         {"point_training":47,"calibration":17,"audit":17})

    def test_trajectory_overlap_rejected(self):
        p=copy.deepcopy(self.protocol)
        p["point_training_files"].append(p["audit_files"][0])
        with self.assertRaises(ValueError): validate_roles(p,self.split,self.universe)

    def test_same_regime_different_trajectory_rejected(self):
        p=copy.deepcopy(self.protocol)
        p["point_training_files"].append("5025_99.h5")
        with self.assertRaisesRegex(ValueError,"Reynolds"):
            validate_roles(p,self.split,self.universe)

    def test_trainer_cannot_include_calibration(self):
        split=copy.deepcopy(self.split)
        split["training_files"].append(self.protocol["calibration_files"][0])
        with self.assertRaises(ValueError): validate_roles(self.protocol,split,self.universe)
