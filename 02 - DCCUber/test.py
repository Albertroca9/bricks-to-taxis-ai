import gymnasium as gym
from TaxiAgent import TaxiAgent

def test(q_table, episodes):
    # Permite testear un agente y visualizar su desempeño
    env = gym.make("Taxi-v4", render_mode="human")

    agent = TaxiAgent(env)
    agent.q_table = q_table.copy()
    agent.exploration_rate = 0.0

    for episode in range(episodes):
        observation, _ = env.reset()
        state = agent.get_state(observation)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = agent.get_action(state, training=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            state = agent.get_state(next_observation)
            total_reward += reward

        print(f"Test episode {episode + 1}: total reward = {total_reward:.2f}" +
              f" {'(Viaje exitoso)' if total_reward > 0.00 else '(Viaje fallido)'}")

    env.close()
