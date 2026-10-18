"""Convergence experiments must measure the requested equation and resolution."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hetfront.studies import Problem, PROBLEMS, errors, standard_study


class StudyTests(unittest.TestCase):
    def test_power_reaches_physical_shock_speed(self):
        # For g=1, theta_L=1, theta_R=0 the shock speed is one for all p.
        # Using theta_L=4 instead distinguishes exponents: speed=4**(1-1/p).
        base = Problem('shock', lambda x: np.ones_like(x), (-3, 3), (-2, 2),
                       (0.,), (4., 0.), .1)
        for power in (1.5, 2., 3.):
            tracker = replace(base, power=power).tracker(.1, 8).run(.1)
            self.assertAlmostEqual(tracker.fronts[0].x(.1), .1 * 4 ** (1 - 1 / power), places=12)

    def test_primitive_sampling_is_nested(self):
        problem = replace(PROBLEMS['line-well-signed'], power=1.5)
        coarse = problem.tracker(.1, 16).run(.2)
        reference = problem.tracker(.025, 64).run(.2)
        e1 = errors(coarse, reference, problem.window, samples=21)
        e2 = errors(coarse, reference, problem.window, samples=41)
        self.assertEqual(e1[0], e2[0])
        # Different summation partitions can perturb shared grid values at roundoff.
        self.assertGreaterEqual(e2[1] + 1e-13, e1[1])

    def test_invalid_study_parameters_fail_before_running(self):
        for args in ({'levels': 1}, {'levels': 2.5}, {'depth': 0},
                     {'samples': 1}, {'coarse_nodes': True}, {'coarse_delta': float('nan')}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                standard_study(PROBLEMS['line-well-signed'], **args)

    def test_cli_keeps_powers_and_reference_metadata_separate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as output:
            subprocess.run([sys.executable, str(root / 'convergence.py'), '--problem',
                            'line-well-signed', '--power', '1.5', '--power', '3',
                            '--levels', '2', '--depth', '1', '--samples', '41',
                            '--max-events', '123456', '--no-plot', '--output', output], cwd=root, check=True,
                           capture_output=True, text=True)
            records = [json.loads(p.read_text()) for p in Path(output).glob('*.json')]
            self.assertEqual({r['power'] for r in records}, {1.5, 3.})
            self.assertEqual(len(list(Path(output).glob('*.csv'))), 2)
            for record in records:
                self.assertEqual(record['reference']['power'], record['power'])
                self.assertEqual(record['reference']['nodes'], 64)
                self.assertEqual(record['reference']['max_events'], 123456)
                for kind in ('delta', 'mesh', 'joint'):
                    self.assertEqual(len(record[kind]), 2)
                    for row in record[kind]:
                        self.assertEqual(row['power'], record['power'])
                        self.assertEqual(row['samples'], 41)
                        self.assertTrue(np.isfinite(row['theta_l1']))
                        self.assertTrue(np.isfinite(row['primitive']))
