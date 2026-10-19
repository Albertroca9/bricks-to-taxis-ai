"""Pruebas del registro experimental de Taxi sin interfaz gráfica."""

from pathlib import Path
import sys
import unittest

import numpy as np

TAXI_DIR = Path(__file__).resolve().parents[1] / "02 - DCCUber"
sys.path.insert(0, str(TAXI_DIR))

from TaxiAgent import TaxiAgent
from train import DCCampeonato, _episode, build_experiments, evaluate_agent, run_experiments, run_training


class Space:
    def __init__(self, n):
        self.n = n


class TruncatedPositiveEnv:
    observation_space = Space(500)
    action_space = Space(6)

    def reset(self, seed=None):
        return 0, {}

    def step(self, action):
        return 1, 20.0, False, True, {}

    def close(self):
        pass


class TaxiTrainingContracts(unittest.TestCase):
    def test_campeonato_trains_650_episodes_and_exports_required_csv(self):
        output = Path(__file__).resolve().parents[1] / "tmp" / "taxi_tests" / "campeonato"
        output.mkdir(parents=True, exist_ok=True)
        agent = DCCampeonato(output_dir=output)
        exported = np.loadtxt(output / "q_table_campeonato.csv", delimiter=",")
        self.assertEqual(agent.n_games, 650)
        self.assertEqual(exported.shape, (500, 7))
        np.testing.assert_array_equal(exported[:, 0], np.arange(500))
        np.testing.assert_allclose(exported[:, 1:], agent.q_table, atol=1e-5)

    def test_truncation_keeps_next_state_value_for_q_learning(self):
        env = TruncatedPositiveEnv()
        agent = TaxiAgent(env)
        agent.exploration_rate = 0.0
        agent.q_table[0, 0] = 1.0
        agent.q_table[1, 0] = 5.0
        _episode(agent, env, observation=0, training=True)
        self.assertAlmostEqual(agent.q_table[0, 0], 3.25)

    def test_training_records_every_episode_and_counts_termination_as_success(self):
        agent, episodes, blocks = run_training(
            3, seed=42, env_factory=TruncatedPositiveEnv, print_every=2
        )
        self.assertEqual(len(episodes), 3)
        self.assertEqual([row["episode"] for row in episodes], [1, 2, 3])
        self.assertTrue(all(row["reward"] == 20 for row in episodes))
        self.assertTrue(all(row["steps"] == 1 for row in episodes))
        self.assertTrue(all(not row["success"] for row in episodes))
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["mean_score"], 20)
        self.assertEqual(agent.n_games, 3)

    def test_evaluation_is_greedy_and_does_not_modify_q_table(self):
        agent, _, _ = run_training(2, seed=7, env_factory=TruncatedPositiveEnv)
        before = agent.q_table.copy()
        evaluation = evaluate_agent(
            agent, episodes=3, seed=100, env_factory=TruncatedPositiveEnv
        )
        np.testing.assert_array_equal(agent.q_table, before)
        self.assertEqual(len(evaluation), 3)
        self.assertTrue(all(not row["success"] for row in evaluation))

    def test_same_seed_reproduces_real_taxi_rewards(self):
        first, first_rows, _ = run_training(8, seed=91)
        second, second_rows, _ = run_training(8, seed=91)
        self.assertEqual(first_rows, second_rows)
        np.testing.assert_array_equal(first.q_table, second.q_table)

    def test_catalog_covers_all_required_conditions(self):
        cases = build_experiments()
        by_group = {}
        for case in cases:
            by_group.setdefault(case["group"], []).append(case)
        self.assertEqual({key: len(value) for key, value in by_group.items()},
                         {"gamma": 5, "learning_rate": 2, "exploration": 2,
                          "duration": 5})
        self.assertEqual([case["episodes"] for case in by_group["duration"]],
                         [500, 2500, 5000, 10000, 20000])
        self.assertTrue(all(case["episodes"] >= 5000 for case in cases
                            if case["group"] != "duration"))
        self.assertEqual([case["agent_config"]["discount_factor"]
                          for case in by_group["gamma"]], [0, .25, .5, .75, 1])

    def test_smoke_experiments_export_tables_figures_and_q_table(self):
        output = Path(__file__).resolve().parents[1] / "tmp" / "taxi_tests" / "experiment_smoke"
        result = run_experiments(output, smoke=True)
        self.assertTrue((output / "SMOKE_ONLY.txt").exists())
        self.assertTrue((output / "summary.csv").exists())
        self.assertTrue((output / "episode_metrics.csv").exists())
        self.assertTrue((output / "block_metrics.csv").exists())
        self.assertTrue((output / "manifest.json").exists())
        self.assertEqual(np.load(output / "q_table_ideal.npy").shape, (500, 6))
        self.assertEqual(len(result["summary"]), 15)
        self.assertEqual(len(list(output.glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
