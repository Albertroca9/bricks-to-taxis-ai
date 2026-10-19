"""Contratos de Q-Learning independientes del render y del entrenamiento largo."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAXI_DIR = ROOT / "02 - DCCUber"
sys.path.insert(0, str(TAXI_DIR))

from TaxiAgent import TaxiAgent


class Space:
    def __init__(self, n):
        self.n = n


class Environment:
    observation_space = Space(500)
    action_space = Space(6)


class TaxiAgentContracts(unittest.TestCase):
    def test_q_table_has_one_row_per_state_and_one_column_per_action(self):
        agent = TaxiAgent(Environment())
        self.assertEqual(agent.q_table.shape, (500, 6))
        np.testing.assert_array_equal(agent.q_table, 0)

    def test_greedy_action_uses_q_values_during_evaluation(self):
        agent = TaxiAgent(Environment())
        agent.q_table[4] = [0, 0, 3, 0, 0, 0]
        self.assertEqual(agent.get_action(4, training=False), 2)

    def test_exploration_uses_a_reproducible_local_generator(self):
        first = TaxiAgent(Environment(), seed=42)
        second = TaxiAgent(Environment(), seed=42)
        first.exploration_rate = second.exploration_rate = 1.0
        first.q_table[4, 5] = second.q_table[4, 5] = 100
        actions_a = [first.get_action(4) for _ in range(20)]
        actions_b = [second.get_action(4) for _ in range(20)]
        self.assertEqual(actions_a, actions_b)
        self.assertGreater(len(set(actions_a)), 1)

    def test_nonterminal_update_bootstraps_and_records_td_error(self):
        agent = TaxiAgent(Environment())
        agent.q_table[1, 0] = 2.0
        agent.q_table[2, 2] = 5.0
        agent.update_q_value(1, 0, 1.0, 2, done=False)
        self.assertAlmostEqual(agent.q_table[1, 0], 2.25)
        self.assertAlmostEqual(agent.training_error[-1], 2.5)

    def test_terminal_update_ignores_next_state_value(self):
        agent = TaxiAgent(Environment())
        agent.q_table[1, 0] = 2.0
        agent.q_table[2, 2] = 5.0
        agent.update_q_value(1, 0, 1.0, 2, done=True)
        self.assertAlmostEqual(agent.q_table[1, 0], 1.9)
        self.assertAlmostEqual(agent.training_error[-1], -1.0)

    def test_csv_export_has_state_plus_six_values_without_header(self):
        agent = TaxiAgent(Environment())
        temp_root = ROOT / "tmp" / "taxi_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        path = temp_root / "q_table_test.csv"
        agent.export_q_table_csv(path)
        exported = np.loadtxt(path, delimiter=",")
        self.assertEqual(exported.shape, (500, 7))
        np.testing.assert_array_equal(exported[:, 0], np.arange(500))


if __name__ == "__main__":
    unittest.main()
