import argparse
import gymnasium as gym
import sys
from stable_baselines3 import PPO
import custom

gym.register(
    id='Custom-v0',
    entry_point='custom:Custom',
    kwargs={'n': 10, 'm': 10, 'num_k': 15, 'max_steps': 100}
)

def main():
    parser = argparse.ArgumentParser(description="Executar uma policy treinada")
    parser.add_argument("--policy", type=str, required=True, help="Ficheiro da policy treinada (zip)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    args = parser.parse_args()

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("CartPole-v0", render_mode="human")
    else:
        # Custom does not accept render_mode in __init__
        env = gym.make("Custom")  # substitui pelo teu ID real

    # Load Policy
    try:
        model = PPO.load(args.policy)
        print(f"Policy '{args.policy}' loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    mode = input("Modo de execução ([c]ontínuo / [s]tep-by-step)? ").strip().lower()
    if mode not in ["c", "s"]:
        print("Opção inválida. A usar modo contínuo por defeito.")
        mode = "c"

    continuar = True
    episodio = 0

    while continuar:
        episodio += 1
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
            
            # display do gym
            if args.gym == "Custom":
                env.unwrapped.draw_env() # display custom
            else:
                env.render() # display standard do gym

            total_reward += reward

            if mode == "s":
                cmd = input("Enter = próximo passo, 'q' = sair deste episódio, 'x' = terminar programa: ").strip().lower()
                if cmd == "x":
                    continuar = False
                    break
                elif cmd == "q":
                    break

        print(f"Episódio {episodio} terminado com recompensa total: {total_reward}")

        if not continuar:
            break

        if mode == "c":
            resp = input("Continuar? (Enter = sim, 'n' = não): ").strip().lower()
            if resp == "n":
                continuar = False

    env.close()

if __name__ == "__main__":
    main()