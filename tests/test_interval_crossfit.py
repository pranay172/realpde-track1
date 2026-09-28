import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from realpde_t1.interval_crossfit import select_crossfit, gather_candidates


class CrossfitTests(unittest.TestCase):
    def test_held_targets_cannot_change_held_choices(self):
        rng=np.random.default_rng(5)
        rewards=rng.random((12,6,4)); groups=np.repeat([1,2,3],4)
        features=np.arange(12)
        for conditional in [False,True]:
            before,_=select_crossfit(rewards,features,groups,conditional)
            perturbed=rewards.copy(); perturbed[groups==1]*=1000
            after,fits=select_crossfit(perturbed,features,groups,conditional)
            np.testing.assert_array_equal(before[groups==1],after[groups==1])
            self.assertNotIn(1,fits[0]["calibration_groups"])
            if conditional: self.assertEqual(fits[0]["threshold"],7.5)

    def test_gather_and_baseline_ties(self):
        values=np.ones((4,6,3)); groups=np.array([1,1,2,2])
        chosen,_=select_crossfit(values,np.arange(4),groups)
        self.assertTrue((chosen==0).all())
        np.testing.assert_array_equal(gather_candidates(values,chosen),values[...,0])

    def test_single_group_rejected(self):
        with self.assertRaises(ValueError):
            select_crossfit(np.ones((2,6,3)),[0,1],[1,1])
