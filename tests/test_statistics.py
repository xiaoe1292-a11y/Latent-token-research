import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
from statistics_utils import paired, pareto


class StatisticsTests(unittest.TestCase):
    def test_every_user_gains_has_exact_unit_difference(self):
        result = paired(dict(ranks=[101] * 8, returned=[50] * 8), dict(ranks=[1] * 8, returned=[100] * 8))
        self.assertEqual((result['delta_recall100'], result['bootstrap_95_low'], result['bootstrap_95_high']), (1, 1, 1))
        self.assertEqual(result['both_full_users'], 0)
        self.assertIsNone(result['delta_recall100_both_full'])

    def test_paired_reordering_and_full_coverage_are_distinguished(self):
        result = paired(dict(ranks=[101, 1, 101, 2], returned=[80, 100, 100, 100]),
                        dict(ranks=[1, 101, 3, 2], returned=[100] * 4))
        self.assertEqual((result['gained_hits100'], result['lost_hits100']), (2, 1))
        self.assertEqual(result['delta_recall100'], .25)
        self.assertEqual(result['gained_from_previously_short'], 1)
        self.assertEqual(result['gained_from_previously_full'], 1)
        self.assertEqual(result['both_full_users'], 3)
        self.assertEqual(result['delta_recall100_both_full'], 0)
        self.assertLess(result['bootstrap_95_low'], .25)
        self.assertGreater(result['bootstrap_95_high'], .25)

    def test_pareto_rejects_slower_worse_points(self):
        rows = [dict(point='A', q=.2, t=5), dict(point='B', q=.3, t=10),
                dict(point='C', q=.1, t=15), dict(point='D', q=.2, t=8)]
        self.assertEqual(pareto(rows, 'q', 't'), ['A', 'B'])

    def test_unpaired_or_empty_populations_are_rejected(self):
        with self.assertRaises(ValueError):
            paired(dict(ranks=[], returned=[]), dict(ranks=[], returned=[]))
        with self.assertRaises(ValueError):
            paired(dict(ranks=[1], returned=[100]), dict(ranks=[1, 2], returned=[100, 100]))


if __name__ == '__main__':
    unittest.main()
