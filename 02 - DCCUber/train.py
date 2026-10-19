"""Entrenamiento tabular de Taxi-v4 y experimentos reproducibles de la tarea."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from TaxiAgent import TaxiAgent


NUM_EPISODES = 10_000
PRINT_EVERY = 100
CAMPEONATO_EPISODES = 650
CAMPEONATO_CONFIG = {
    "discount_factor": 0.99,
    "learning_rate": 0.8,
    "epsilon_decay_rate": 0.005,
    "min_epsilon": 0.05,
}


def make_env():
    """Crea Taxi sin render para entrenar y evaluar con rapidez."""
    return gym.make("Taxi-v4")


def _episode(agent, env, observation, training):
    state = agent.get_state(observation)
    total_reward = 0.0
    steps = 0
    while True:
        action = agent.get_action(state, training=training)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.get_state(next_observation)
        if training:
            # El límite de pasos trunca el episodio, pero no hace terminal al estado.
            agent.update_q_value(state, action, reward, next_state, terminated)
        total_reward += reward
        steps += 1
        state = next_state
        if terminated or truncated:
            return total_reward, steps, bool(terminated)


def _block_summary(rows):
    return {
        "episode": rows[-1]["episode"],
        "mean_score": float(np.mean([row["reward"] for row in rows])),
        "best_score": float(np.max([row["reward"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
    }


def run_training(episodes, seed=42, agent_config=None, env_factory=make_env,
                 print_every=PRINT_EVERY):
    """Entrena desde una tabla nueva y devuelve agente, episodios y bloques."""
    if episodes < 1 or print_every < 1:
        raise ValueError("episodes y print_every deben ser positivos")
    env = env_factory()
    try:
        agent = TaxiAgent(env, seed=seed, **(agent_config or {}))
        episode_rows = []
        block_rows = []
        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode)
            reward, steps, success = _episode(agent, env, observation, training=True)
            agent.n_games += 1
            agent.decay_exploration_rate()
            episode_rows.append({
                "episode": episode,
                "reward": float(reward),
                "steps": steps,
                "success": success,
                "epsilon": float(agent.exploration_rate),
            })
            if episode % print_every == 0 or episode == episodes:
                block_rows.append(_block_summary(episode_rows[-min(print_every, episode % print_every or print_every):]))
        return agent, episode_rows, block_rows
    finally:
        env.close()


def evaluate_agent(agent, episodes=100, seed=100_000, env_factory=make_env):
    """Evalúa la política codiciosa sin modificar la Q-table."""
    if episodes < 1:
        raise ValueError("episodes debe ser positivo")
    env = env_factory()
    try:
        rows = []
        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode)
            reward, steps, success = _episode(agent, env, observation, training=False)
            rows.append({"episode": episode, "reward": float(reward),
                         "steps": steps, "success": success})
        return rows
    finally:
        env.close()


def plot_learning_curve(mean_scores, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(mean_scores) + 1) * PRINT_EVERY, mean_scores)
    plt.xlabel("Episodios de entrenamiento")
    plt.ylabel("Mean score (recompensa media del bloque)")
    plt.title("Aprendizaje de Taxi con Q-Learning")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(save_npy_path, save_csv_path, save_png_path):
    """Entrada conservada para el main.py original."""
    agent, _, blocks = run_training(NUM_EPISODES)
    for row in blocks:
        print(f"Episode {row['episode']:6d} | mean score: {row['mean_score']:8.2f} | "
              f"best score: {row['best_score']:8.2f} | "
              f"% viajes exitosos: {100 * row['success_rate']:6.2f}%")
    np.save(save_npy_path, agent.q_table)
    agent.export_q_table_csv(save_csv_path)
    plot_learning_curve([row["mean_score"] for row in blocks], save_png_path)
    return agent


def build_experiments(smoke=False):
    """Configuraciones obligatorias, fijadas antes de observar resultados."""
    cases = []
    for gamma in [0, .25, .5, .75, 1]:
        cases.append({"group": "gamma", "scenario": f"gamma_{gamma:g}",
                      "episodes": 5000, "agent_config": {"discount_factor": gamma}})
    for rate in [.01, .7]:
        cases.append({"group": "learning_rate", "scenario": f"alpha_{rate:g}",
                      "episodes": 5000, "agent_config": {"learning_rate": rate}})
    cases.extend([
        {"group": "exploration", "scenario": "decay_prematuro",
         "episodes": 5000, "agent_config": {"epsilon_decay_rate": .05}},
        {"group": "exploration", "scenario": "exploracion_eterna",
         "episodes": 5000,
         "agent_config": {"max_epsilon": .8, "min_epsilon": .8}},
    ])
    for episodes in [500, 2500, 5000, 10000, 20000]:
        cases.append({"group": "duration", "scenario": f"episodes_{episodes}",
                      "episodes": episodes, "agent_config": {}})
    if smoke:
        return [{**case, "episodes": 10} for case in cases]
    return cases


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Sin filas para {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def _plot_group(output_dir, group, block_rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = dict.fromkeys(row["scenario"] for row in block_rows
                              if row["group"] == group)
    for scenario in scenarios:
        rows = [row for row in block_rows if row["scenario"] == scenario]
        ax.plot([row["episode"] for row in rows],
                [row["mean_score"] for row in rows], label=scenario)
    ax.set(xlabel="Episodios", ylabel="Mean score (último bloque de 100)",
           title=f"Taxi-v4: {group}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{group}.png", dpi=150)
    plt.close(fig)


def run_experiments(output_dir, smoke=False):
    """Ejecuta las condiciones obligatorias y guarda evidencia por escenario."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_experiments(smoke=smoke)
    training_seed = 42
    validation_seed = 100_000
    final_seed = 200_000
    validation_episodes = 3 if smoke else 100
    episode_rows = []
    block_rows = []
    evaluation_rows = []
    summary_rows = []

    def execute(case, seed, evaluation_seed):
        print(f"Entrenando {case['scenario']}: {case['episodes']} episodios", flush=True)
        agent, episodes, blocks = run_training(
            case["episodes"], seed=seed, agent_config=case["agent_config"]
        )
        evaluation = evaluate_agent(
            agent, episodes=validation_episodes, seed=evaluation_seed
        )
        prefix = {"group": case["group"], "scenario": case["scenario"]}
        episode_rows.extend([{**prefix, **row} for row in episodes])
        block_rows.extend([{**prefix, **row} for row in blocks])
        evaluation_rows.extend([{**prefix, **row} for row in evaluation])
        summary = {
            **prefix,
            "episodes": case["episodes"],
            "agent_config": json.dumps(case["agent_config"], sort_keys=True),
            "final_mean_score": blocks[-1]["mean_score"],
            "max_score": max(row["reward"] for row in episodes),
            "train_success_rate_last100": blocks[-1]["success_rate"],
            "eval_mean_score": _mean(evaluation, "reward"),
            "eval_success_rate": _mean(evaluation, "success"),
            "eval_mean_steps": _mean(evaluation, "steps"),
        }
        summary_rows.append(summary)
        print(f"  Entrenamiento final {summary['final_mean_score']:.2f}; "
              f"evaluación codiciosa {summary['eval_mean_score']:.2f}; "
              f"éxito {summary['eval_success_rate']:.1%}", flush=True)
        return agent, summary

    for case in cases:
        execute(case, training_seed, validation_seed)

    def best(group):
        options = [row for row in summary_rows if row["group"] == group]
        return max(options, key=lambda row: (
            row["eval_success_rate"], row["eval_mean_score"], -row["episodes"]
        ))

    selected_gamma = best("gamma")
    selected_rate = best("learning_rate")
    selected_duration = best("duration")
    ideal_config = {
        "discount_factor": json.loads(selected_gamma["agent_config"])["discount_factor"],
        "learning_rate": json.loads(selected_rate["agent_config"])["learning_rate"],
    }
    ideal_case = {
        "group": "ideal", "scenario": "agente_ideal",
        "episodes": 10 if smoke else selected_duration["episodes"],
        "agent_config": ideal_config,
    }
    ideal_agent, ideal_summary = execute(ideal_case, 31415, final_seed)
    np.save(output_dir / "q_table_ideal.npy", ideal_agent.q_table)
    ideal_agent.export_q_table_csv(output_dir / "q_table_ideal.csv")
    _write_csv(output_dir / "episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "block_metrics.csv", block_rows)
    _write_csv(output_dir / "evaluation_metrics.csv", evaluation_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "learning_rate_blocks.csv",
               [row for row in block_rows if row["group"] == "learning_rate"])
    _write_csv(output_dir / "duration_summary.csv",
               [row for row in summary_rows if row["group"] == "duration"])
    for group in ["gamma", "learning_rate", "exploration", "duration", "ideal"]:
        _plot_group(output_dir, group, block_rows)
    manifest = {
        "smoke": smoke,
        "environment": "Taxi-v4",
        "training_seed": training_seed,
        "validation_seed": validation_seed,
        "final_seed": final_seed,
        "validation_episodes": validation_episodes,
        "gymnasium_version": gym.__version__,
        "numpy_version": np.__version__,
        "conditions": cases,
        "ideal_selection": {
            "gamma_from": selected_gamma["scenario"],
            "learning_rate_from": selected_rate["scenario"],
            "duration_from": selected_duration["scenario"],
            "ideal_case": ideal_case,
            "final_evaluation": ideal_summary,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if smoke:
        (output_dir / "SMOKE_ONLY.txt").write_text(
            "Ejecución de verificación reducida; no usar sus métricas en la entrega.\n",
            encoding="utf-8",
        )
    return {"summary": summary_rows, "manifest": manifest}


def DCCampeonato(output_dir=None):
    """Entrena 650 episodios y exporta la tabla exigida por el campeonato.

    Sin output_dir, el CSV se guarda en el directorio desde el que se ejecuta.
    """
    destination = Path.cwd() if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    agent, _, blocks = run_training(
        CAMPEONATO_EPISODES, seed=42, agent_config=CAMPEONATO_CONFIG
    )
    csv_path = destination / "q_table_campeonato.csv"
    agent.export_q_table_csv(csv_path)
    print(f"DCCampeonato: {agent.n_games} episodios; "
          f"mean score final: {blocks[-1]['mean_score']:.2f}; CSV: {csv_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Taxi-v4")
    parser.add_argument("--experiments", action="store_true",
                        help="Ejecuta los experimentos completos")
    parser.add_argument("--smoke", action="store_true",
                        help="Ejecuta una comprobación breve sin valor académico")
    parser.add_argument("--campeonato", action="store_true",
                        help="Entrena 650 episodios y exporta q_table_campeonato.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Carpeta de salida; por defecto resultados/<UTC timestamp>")
    args = parser.parse_args()
    if sum([args.experiments, args.smoke, args.campeonato]) != 1:
        parser.error("Elige exactamente una opción: --experiments, --smoke o --campeonato")
    if args.campeonato:
        DCCampeonato(output_dir=args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = args.output or Path(__file__).resolve().parent / "resultados" / timestamp
        run_experiments(destination, smoke=args.smoke)
        print(f"Resultados guardados en {destination}")
