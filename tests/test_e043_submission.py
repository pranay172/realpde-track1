import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location("e043_test",REPO/"submission/e043/submission.py")
        self.module=importlib.util.module_from_spec(spec)
        calibration={"threshold":.01,"scales":[[[.5,1],[.75,1.25],[1.5,2]],[[1,2],[1.5,1],[.75,.5]]],
                     "horizons":[[0,4],[4,10],[10,20]],"sigma_global":.0563870259}
        with patch("builtins.open",mock_open(read_data=json.dumps(calibration))): spec.loader.exec_module(self.module)
        self.x=np.random.default_rng(7).normal(0,.02,(2,20,32,64,3)).astype(np.float32)

    def test_bounds_batch_permutation_and_pressure(self):
        point=self.x.copy(); point[...,2]=0
        lo,hi=self.module.conditional_bounds(point,self.x)
        for i in range(2):
            a,b=self.module.conditional_bounds(point[i:i+1],self.x[i:i+1])
            np.testing.assert_array_equal(a,lo[i:i+1]); np.testing.assert_array_equal(b,hi[i:i+1])
        self.assertTrue((lo<=point).all() and (point<=hi).all())
        self.assertTrue((lo[...,2]==0).all() and (hi[...,2]==0).all())

    def test_prediction_normalization_and_persistence(self):
        self.module.STATE.update(model=torch.nn.Identity(),device=torch.device("cpu"),
            normalizer={"mean_input":[0,0,0],"std_input":[1,1,1],"mean_target":[0,0,0],"std_target":[1,1,1]})
        result=self.module.predict(self.x)
        expected=self.x.copy(); expected[:,:4]=self.x[:,-1:]; expected[...,2]=0
        np.testing.assert_array_equal(result["prediction"],expected)

    def test_invalid_input_rejected(self):
        for bad in [self.x[:0],self.x[0],np.full_like(self.x,np.nan)]:
            with self.assertRaises(ValueError): self.module.predict(bad)
