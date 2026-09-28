import sys
from pathlib import Path
import unittest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "realpde_t1_starting_kit_v9")]
import scoring
from realpde_t1.sps_diagnostics import sps_components, summarize_sps


class SPSDiagnosticsTests(unittest.TestCase):
    def test_matches_official_with_unequal_masks_and_chunking(self):
        rng = np.random.default_rng(5)
        t = rng.normal(.1, .03, (3,20,32,64,3)).astype(np.float32)
        p = t + rng.normal(0, .02, t.shape).astype(np.float32)
        t[0,:,:16,:,:2] = 0
        p[...,2] = t[...,2] = 0
        lo, hi = p - .018, p + .018
        whole = sps_components(p,t,lo,hi,scoring)
        result = summarize_sps(whole)
        score, coverage = scoring.aggregate_sps(p,t,2,lo,hi)
        self.assertAlmostEqual(result["sps_score"], 100*score, places=5)
        self.assertAlmostEqual(result["coverage_percent"], 100*coverage, places=5)
        chunks = [sps_components(p[i:i+1],t[i:i+1],lo[i:i+1],hi[i:i+1],scoring) for i in range(3)]
        joined = {k:np.concatenate([c[k] for c in chunks]) for k in whole}
        self.assertAlmostEqual(summarize_sps(joined)["sps_score"],100*score,places=5)
        self.assertGreaterEqual(result["symmetric_oracle_sps"] + 1e-5, result["sps_score"])
        self.assertGreaterEqual(result["accuracy_ceiling_sps"],result["symmetric_oracle_sps"])

    def test_perfect_prediction_ceiling_and_zero_mask(self):
        p = np.full((1,20,32,64,3), .1, dtype=np.float32)
        p[...,2] = 0
        r = summarize_sps(sps_components(p,p,p,p,scoring))
        self.assertEqual(r["sps_score"],100)
        self.assertEqual(r["symmetric_oracle_sps"],100)
        z = np.zeros_like(p)
        self.assertEqual(summarize_sps(sps_components(z,z,z,z,scoring)),{"scored_elements":0})
