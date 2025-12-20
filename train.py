import argparse
import pickle
import gym
import seals  # biblioteca para ambientes imitation
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.data import rollout

def main():
    parser = argparse.ArgumentParser(description="Treino por Aprendizagem por Imitação")
    parser.add_argument("--file", type=str, required=True, help="Ficheiro com demonstrações (pkl)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algoritmo de aprendizagem por imitação")
    parser.add_argument("--output", type=str, required=True, help="Ficheiro de output da política treinada")
    args = parser.parse_args()

    # Carregar demonstrações
    with open(args.file, "rb") as f:
        demos = pickle.load(f)

    # Converter para formato Transitions (se necessário)
    transitions = Transitions(
        obs=demos["obs"],
        acts=demos["acts"],
        infos=demos.get("infos", [{}] * len(demos["obs"])),
        next_obs=demos["next_obs"],
        dones=demos["dones"],
    )

    # Selecionar ambiente
    if args.gym == "CartPole":
        env = gym.make("seals/CartPole-v0")
    else:
        # Placeholder para ambiente customizado
        env = gym.make("CartPole-v1")  # substitua pelo seu ambiente custom
    
    # Selecionar algoritmo
    def select_algorithm_for_imitation():
        
        if args.algorithm == "BC":
            # Treino para BC
            # https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html 

            bc_trainer = BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions)
            
            bc_trainer.train(n_epochs=10)
            bc_trainer.policy.save(args.output)
            
            return bc_trainer

        elif args.algorithm == "GAIL":
            # Política base (PPO)
            
            # Treino para GAIL
            # https://imitation.readthedocs.io/en/latest/tutorials/3_train_gail.html

            policy = PPO("MlpPolicy", env, verbose=1)

            gail_trainer = GAIL(
                demonstrations=transitions,
                gen_algo=policy,
                reward_net=None)  # reward_net pode ser definido
            
            num_passos = 10000

            gail_trainer.train(num_passos)  # número de passos
            policy.save(args.output)
            
            return policy

    new_policy = select_algorithm_for_imitation()
    # Treina uma policy P para aproximar D (Demonstrações), utilizando A (Algoritmo) em G (nome do ginásio); 

if __name__ == "__main__":
    main()
