import argparse
import pickle
import gymnasium as gym
import seals  # biblioteca para ambientes imitation
import numpy as np
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Gerar demonstrações de um ginásio")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--episodes", type=int, required=True, help="Número de episódios de demonstração")
    parser.add_argument("--file", type=str, required=True, help="Ficheiro de output para guardar demonstrações")
    parser.add_argument("--use_ppo_huggingface", action="store_true", help="Usar policy PPO pré-treinada do HuggingFace (apenas CartPole)")
    args = parser.parse_args()

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("seals/CartPole-v0")
    else:
        # Placeholder para ambiente customizado
        env = gym.make("CartPole-v1")  # substitua pelo seu ambiente custom

    # Selecionar policy
    if args.use_ppo_huggingface and args.gym == "CartPole":
        # Carregar modelo PPO pré-treinado do HuggingFace
        model = PPO.load("ppo-CartPole")  # assumindo que já foi descarregado
        use_model = True
    else:
        model = None
        use_model = False

    # Estrutura para guardar demonstrações
    demos = {
        "obs": [],
        "acts": [],
        "next_obs": [],
        "dones": [],
        "infos": []
    }

    # Gerar episódios
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # política aleatória

            next_obs, reward, done, info = env.step(action)

            demos["obs"].append(obs)
            demos["acts"].append(action)
            demos["next_obs"].append(next_obs)
            demos["dones"].append(done)
            demos["infos"].append(info)

            obs = next_obs

    # Guardar em ficheiro
    with open(args.file, "wb") as f:
        pickle.dump(demos, f)

    print(f"Demonstrações guardadas em {args.file}")

if __name__ == "__main__":
    main()
