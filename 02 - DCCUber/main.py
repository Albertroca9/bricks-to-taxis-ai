from train import train
from test import test

# Parámetros de guardado y testeo
SAVE_CSV_PATH = "q_table.csv"
SAVE_NPY_PATH = "q_table.npy"
SAVE_PNG_PATH = "training_curve.png"
TEST_EPISODES = 5

if __name__ == "__main__":
    # Se entrena el agente y luego se testea usando la Q-table obtenida
    trained_agent = train(SAVE_NPY_PATH, SAVE_CSV_PATH, SAVE_PNG_PATH)
    input("Entrenamiento completo. Presiona Enter para iniciar las pruebas...")
    test(trained_agent.q_table, TEST_EPISODES)
