import argparse
import gymnasium as gym
import seals  # biblioteca para ambientes imitation
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Executar uma policy treinada")
    parser.add_argument("--policy", type=str, required=True, help="Ficheiro da policy treinada (zip)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    args = parser.parse_args()

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("seals/CartPole-v0")
    else:
        # Placeholder para ambiente customizado
        env = gym.make("CartPole-v1")  # substitua pelo seu ambiente custom

    # Carregar policy
    model = PPO.load(args.policy)

    # Executar episódios

    num_episodios = 5

    for ep in range(num_episodios):  # correr n episódios
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
        print(f"Episódio {ep+1} terminado com recompensa total: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
