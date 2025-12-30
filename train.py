import argparse 
import numpy as np

# import custom_enviornment # Biblioteca com um novo ambiente

from imitation.data import rollout
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.bc import BC

from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy


def main():

    # Argumento de entrada
    parser = argparse.ArgumentParser(description="Treino por Aprendizagem por Imitação")
    parser.add_argument("--file", type=str, required=True, help="Ficheiro com demonstrações (pkl)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algoritmo de aprendizagem por imitação")
    parser.add_argument("--output", type=str, required=True, help="Ficheiro de output da política treinada")
    args = parser.parse_args()

    def get_ambiente(type_gym):
        if type_gym == "CartPole":
            return "seals:seals/CartPole-v0"
        else:
            # Com o mesmo id que foi usado para o registo do custom
            return "custom/custom_enviornment" 

    def load_env(seed, type_env, type_algorithm):

        if type_algorithm == "BC":
            return make_vec_env(
                type_env,
                rng=np.random.default_rng(seed),
                post_wrappers=[
                    lambda env, _: RolloutInfoWrapper(env)
                ],  # needed for computing rollouts later
            )

        elif type_algorithm == "GAIL":
            return make_vec_env(
                type_env,
                rng=np.random.default_rng(seed),
                n_envs=8,
                post_wrappers=[
                    lambda env, _: RolloutInfoWrapper(env)
                ],  # needed for computing rollouts later
        )
        else:
            return None

    """ 
    Selecionar o tipo de ambiente (CartPole ou Custom)
    Criar o ambiente
    Descarregar as demonstrações do expert
    Treino da policy e guarda no ficherio output
    """

    SEED = 42

    # Selecionar o tipo de ambiente
    type_env = get_ambiente(args.gym)

    # Carregar o ambiente (no GAIL, adicionar n_envs=8,)
    env = load_env(SEED, type_env, args.algorithm)

    # Descarregar as demonstrações (para ambos os algoritmos)
    #expert = load_policy(env_name=type_env, venv=env, path=args.file)
    
    # Teste da policy (Mudar quando tiver o ambiente custom)
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals/CartPole-v0",
        venv=env,
    )

    if args.algorithm == "BC":
        reward, _ = evaluate_policy(expert, env, 10)
        print(reward)

    # Para o GAIL, min_episodes = 60
    rng = np.random.default_rng(SEED)
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )

    if args.algorithm == "BC":
        
        print(
            f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
            After flattening, this list is turned into a {type(rollout.flatten_trajectories(rollouts))} object containing {len(rollout.flatten_trajectories(rollouts))} transitions.
            The transitions object contains arrays for: {', '.join(rollout.flatten_trajectories(rollouts).__dict__.keys())}."
            """
            )

        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=rollout.flatten_trajectories(rollouts),
            rng=rng)
        
        reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 5)

        bc_trainer.train(n_epochs=5)
                
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 5)

        print(f"Reward before training: {reward_before_training}")
        print(f"Reward after training: {reward_after_training}")

        save_policy = PPO( policy=bc_trainer.policy.__class__, env=env, verbose=0, ) 
        save_policy.policy.load_state_dict(bc_trainer.policy.state_dict()) 
        save_policy.save(args.output)
        
        env.close()
        
    elif args.algorithm == "GAIL":
        
        learner = PPO(
            env=env, 
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0004,
            gamma=0.95,
            n_epochs=5,
            seed=SEED,
        )

        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
            normalize_input_layer=RunningNorm,
        )

        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )

        env.seed(SEED)
        learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True)

        num_passos = 200000

        gail_trainer.train(num_passos)  # número de passos
        learner.save(args.output)

        env.seed(SEED)
        learner_rewards_after_training, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)

        print(
            "Rewards before training:", np.mean(learner_rewards_before_training),
            "+/-", np.std(learner_rewards_before_training),)

        print(
            "Rewards after training:", np.mean(learner_rewards_after_training),
            "+/-", np.std(learner_rewards_after_training),
        )
        env.close()
            
if __name__ == "__main__":
    main()
