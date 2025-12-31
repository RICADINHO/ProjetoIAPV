import argparse
import gymnasium as gym
import seals
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description="Executar uma policy treinada")
    parser.add_argument("--policy", type=str, required=True, help="Ficheiro da policy treinada (zip)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    args = parser.parse_args()

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("seals:seals/CartPole-v0", render_mode="human")
    else:
        env = gym.make("Custom", render_mode="human")  # substitui pelo teu ID real

    # Carregar policy
    model = PPO.load(args.policy)

    # Escolha do utilizador: step-by-step ou contínuo
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

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward

            if mode == "s":
                
                cmd = input("Enter = próximo passo, 'q' = sair deste episódio, 'x' = terminar programa: ").strip().lower()
                if cmd == "x":
                    continuar = False
                    break
                elif cmd == "q":
                    break
                # qualquer outra tecla ou Enter continua passo a passo

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
