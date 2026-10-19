import numpy as np

LR = 0.10
DISCOUNT_RATE = 0.70
MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.0015


class TaxiAgent:
    # Clase de agente Q-Learning para el entorno Taxi-v4.

    def __init__(self, env, learning_rate=LR, discount_factor=DISCOUNT_RATE,
                 max_epsilon=MAX_EXPLORATION_RATE, min_epsilon=MIN_EXPLORATION_RATE,
                 epsilon_decay_rate=EXPLORATION_DECAY_RATE, seed=None):
        self.env = env

        # Inicializamos los hiperparámetros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.n_games = 0
        self.exploration_rate = self.max_epsilon
        self.training_error = []
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def get_state(self, observation):
        # El observation ya viene como el índice de estado (0-499)
        return int(observation)

    def get_action(self, state, training=True):
        if training and self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))
    
    def update_q_value(self, state, action, reward, next_state, done):
        # Actualiza la Q-table usando la fórmula de Q-Learning

        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        self.training_error.append(td_error)
    
    def decay_exploration_rate(self):
        # Actualiza el exploration rate una vez por episodio
        self.exploration_rate = (self.min_epsilon +
                            (self.max_epsilon - self.min_epsilon)
                            * np.exp(-self.epsilon_decay_rate * self.n_games))

    def export_q_table_csv(self, filepath):
        # Exporta la Q-table obtenida a un archivo CSV
        # Cada fila: [estado, Q(estado, accion_0), ..., Q(estado, accion_5)]
        rows = []
        for state in range(self.n_states):
            rows.append([state] + self.q_table[state].tolist())
        np.savetxt(filepath, np.asarray(rows, dtype=np.float32), delimiter=",", fmt="%.6f")
