import argparse
import gymnasium as gym
import csv
from stable_baselines3 import PPO
import custom

gym.register(
    id='Custom-v0',
    entry_point='custom:Custom',
    kwargs={'n': 10, 'm': 10, 'num_k': 15, 'max_steps': 100}
)

def main():
    parser = argparse.ArgumentParser(description="Executar uma policy treinada por 100 episódios")
    parser.add_argument("--policy", type=str, required=True, help="Ficheiro da policy treinada (zip)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--out", type=str, default="resultados.csv", help="Ficheiro CSV de saída")
    args = parser.parse_args()

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("CartPole-v0", render_mode="human")
    else:
        env = gym.make("Custom")

    # Load Policy
    try:
        model = PPO.load(args.policy)
        print(f"Policy '{args.policy}' loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Criar ficheiro CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "recompensa_total"])

        # Executar 100 episódios
        for episodio in range(1, 101):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            print(f"\n--- Episódio {episodio} ---")

            if args.gym == "Custom":
                env.unwrapped.draw_env()

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                if args.gym == "Custom":
                    env.unwrapped.draw_env()
                else:
                    env.render()

                total_reward += reward

            print(f"Episódio {episodio} terminado com recompensa total: {total_reward}")

            # Guardar no CSV
            writer.writerow([episodio, total_reward])

    env.close()
    print(f"\nResultados guardados em: {args.out}")

if __name__ == "__main__":
    main()
