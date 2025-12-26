import argparse
import pickle

#import seals  # biblioteca para ambientes imitation
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.data import rollout

import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper

def main():

    # Argumento de entrada
    parser = argparse.ArgumentParser(description="Treino por Aprendizagem por Imitação")
    parser.add_argument("--file", type=str, required=True, help="Ficheiro com demonstrações (pkl)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algoritmo de aprendizagem por imitação")
    parser.add_argument("--output", type=str, required=True, help="Ficheiro de output da política treinada")
    args = parser.parse_args()

    """ 
    Selecionar o tipo de ambiente (CartPole ou Custom)
    Criar o ambiente
    Descarregar as demonstrações do expert
    Treino da policy e guarda no ficherio output
    """

    # Selecionar o tipo de ambiente
    def get_ambiente():
        if args.gym == "CartPole":
            return "seals/CartPole-v0"
        else:
            # Placeholder para ambiente customizado
            return "Custom"  # substitua pelo seu ambiente custom
   
    type_env = get_ambiente

    # Carregar o ambiente (no GAIL, adicionar n_envs=8,)
    env = make_vec_env(
        "seals:"+type_env,
        rng=np.random.default_rng(),
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )

    # Descarregar as demonstrações (para ambos os algoritmos)
    expert = load_policy(
        env_name=type_env,
        venv=env,
        path=args.file
    )

    # Para o GAIL, adicionar seed no rng = 42 e min_episodes = 60
    rng = np.random.default_rng()
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )

    # Selecionar algoritmo
    def select_algorithm_for_imitation():
        
        if args.algorithm == "BC":
            
            bc_trainer = BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=rollout.flatten_trajectories(rollouts),
                rng=rng)
            
            bc_trainer.train(n_epochs=10)
            bc_trainer.policy.save(args.output)
            

        elif args.algorithm == "GAIL":

            policy = PPO("MlpPolicy", env, verbose=1)

            gail_trainer = GAIL(
                demonstrations=transitions,
                gen_algo=policy,
                reward_net=None)  # reward_net pode ser definido
            
            num_passos = 10000

            gail_trainer.train(num_passos)  # número de passos
            policy.save(args.output)
            
            return policy

    select_algorithm_for_imitation()

if __name__ == "__main__":
    main()
