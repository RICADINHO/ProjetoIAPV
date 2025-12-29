import argparse
import gymnasium as gym
import seals  # biblioteca para ambientes imitation
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Executar uma policy treinada")
    parser.add_argument("--policy", type=str, required=True, help="Ficheiro da policy treinada (zip)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    args = parser.parse_args()

    """
    Receber ficheiro com a policy e o nome do ambiente que foi treinado a policy
    Abrir o ambiente em mode de visualização
    Executa Policy no ambiente passo a passo ou em modo contínuo, de acordo com a 
    escolha do utilizador e enquanto o pretender
    """

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("seals/CartPole-v0")
    else:
        # Placeholder para ambiente customizado
        env = gym.make("Custom")  # substitua pelo seu ambiente custom

    # Melhorar
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
