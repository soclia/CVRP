import unittest
import numpy as np
from paired_statistics import paired_summary


class PairedStatisticsTests(unittest.TestCase):
    def test_constant_paired_difference_and_reproducible_interval(self):
        result = paired_summary([98]*16, [100]*16, [100]*16, bootstrap_repeats=200, seed=3)
        self.assertAlmostEqual(result['mean_difference_pp'], -2.0)
        np.testing.assert_allclose(result['confidence_interval_95_pp'], [-2,-2])
        self.assertLess(result['wilcoxon_p_value'], 0.05)
        self.assertEqual(result, paired_summary([98]*16, [100]*16, [100]*16, 200, 3))

    def test_seed_aggregation_keeps_instance_sample_size(self):
        result = paired_summary([[9,10],[11,8]], [11,10], [10,10], 100, 3)
        self.assertEqual(result['instances'], 2)
        self.assertEqual(result['ours_models'], 2)
        self.assertAlmostEqual(result['mean_difference_pp'], -10)

    def test_zero_differences_and_invalid_reference(self):
        result = paired_summary([2,3], [2,3], [2,3], 100, 2)
        self.assertEqual(result['wilcoxon_p_value'], 1.0)
        for reference in ([1,0], [1], [1,float('nan')]):
            with self.assertRaises(ValueError):
                paired_summary([2,3], [2,3], reference, 100, 2)
