import argparse
import numpy as np
import gymnasium as gym

from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.bc import BC

from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.evaluation import evaluate_policy

import pickle
from imitation.data import types
from imitation.data import rollout as rollout_utils


def returns_from_demos(demos):
    """
    Calcula média e desvio padrão dos retornos das demonstrações.
    Aceita:
      - lista de objetos Trajectory (com .rews ou comprimento de obs)
      - lista de dicts com keys 'rews' ou 'obs'
    Para ambientes como CartPole, se não houver 'rews', assume recompensa 1 por passo.
    """
    episode_returns = []
    for traj in demos:
        # Trajectory object (imitation.types.Trajectory)
        if hasattr(traj, "rews") or hasattr(traj, "obs"):
            # objeto Trajectory
            rews = getattr(traj, "rews", None)
            if rews is None:
                # se não houver rews, assumir 1 por passo
                obs = getattr(traj, "obs", None)
                if obs is not None:
                    episode_returns.append(len(obs))
                else:
                    episode_returns.append(0.0)
            else:
                episode_returns.append(float(np.sum(rews)))
        # dict-like trajectory (com 'rews' ou 'obs')
        elif isinstance(traj, dict):
            if "rews" in traj and traj["rews"] is not None:
                episode_returns.append(float(np.sum(traj["rews"])))
            elif "obs" in traj and traj["obs"] is not None:
                # assumir recompensa 1 por passo se rews ausentes
                episode_returns.append(len(traj["obs"]))
            else:
                episode_returns.append(0.0)
        else:
            # fallback: tentar iterar por passos com 'reward' campo
            try:
                total = 0.0
                for step in traj:
                    total += step.get("reward", 0.0)
                episode_returns.append(total)
            except Exception:
                episode_returns.append(0.0)

    if len(episode_returns) == 0:
        return 0.0, 0.0
    return float(np.mean(episode_returns)), float(np.std(episode_returns))

# Estruturação dos pares de estado-ação
def trajs_from_imitation_trajectories(expert_list):
   
    trajs_dict = []
    traj_objs = []
    for traj in expert_list:
        obs = np.asarray(traj.obs)
        acts = np.asarray(traj.acts)

        # Se não houver rews, assumir 1 por passo (ex.: CartPole)
        rews = np.asarray(getattr(traj, "rews", np.ones(len(acts), dtype=float)))
        dones = np.asarray(getattr(traj, "dones", np.array([False] * len(acts))))
        infos = getattr(traj, "infos", [{}] * len(acts))

        trajs_dict.append({"obs": obs, "acts": acts, "rews": rews, "dones": dones, "infos": infos})

        # types.Trajectory normalmente aceita obs, acts, infos, terminal
        traj_obj = types.Trajectory(
            obs=obs,
            acts=acts,
            infos=np.asarray(infos, dtype=object),
            terminal=bool(getattr(traj, "terminal", True)),
        )
        traj_objs.append(traj_obj)

    return trajs_dict, traj_objs

# Carrega demonstrações salvas em um arquivo .pkl
def load_demonstrations(filename):
     
    try:
        with open(filename, "rb") as f:
            demos = pickle.load(f)
        print(f"[OK] {len(demos)} demonstrações carregadas de '{filename}'")
        return demos
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{filename}' não encontrado.")
        return []
    except Exception as e:
        print(f"[ERRO] Falha ao carregar: {e}")
        return []

# Selecionar o ambiente que o agente vai treinar a poliica
def get_ambiente(type_gym):
    if type_gym == "CartPole":
        return "CartPole-v0"
    else:
        return "Custom-v0"

# Carregar o ambiente escolhido com um algoritmo de imitação
def load_env(seed, type_env, type_algorithm):
    
    if type_algorithm == "BC":
        return make_vec_env(
            type_env,
            rng=np.random.default_rng(seed),
            post_wrappers=[
                lambda env, _: RolloutInfoWrapper(env)
            ],
        )
    elif type_algorithm == "GAIL":
        return make_vec_env(
            type_env,
            rng=np.random.default_rng(seed),
            n_envs=8,
            post_wrappers=[
                lambda env, _: RolloutInfoWrapper(env)
            ],
        )
    else:
        return None

def main():
    # Argumento de entrada
    parser = argparse.ArgumentParser(description="Treino por Aprendizagem por Imitação")
    parser.add_argument("--file", type=str, required=True, help="Ficheiro com demonstrações (pkl)")
    parser.add_argument("--gym", type=str, required=True, choices=["CartPole", "Custom"], help="Nome do ginásio")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algoritmo de aprendizagem por imitação")
    parser.add_argument("--output", type=str, required=True, help="Ficheiro de output da política treinada")
    args = parser.parse_args()

    # Registo do ambiente Custom
    gym.register(
        id='Custom-v0', entry_point='custom:Custom', kwargs={'n': 10, 'm': 10, 'num_k': 15, 'max_steps': 100}
    )

    SEED = 42
    rng = np.random.default_rng(SEED)

    # Selecionar o tipo de ambiente
    type_env = get_ambiente(args.gym)

    # Carregar o ambiente
    env = load_env(SEED, type_env, args.algorithm)

    # Carregar demonstrações
    expert = load_demonstrations(args.file)
    
    print("DEBUG: tipo de expert:", type(expert))
    if len(expert) > 0:
        print("DEBUG: exemplo expert[0]:", expert[0])
    else:
        print("DEBUG: expert está vazio")

    # Converter demonstrações (objetos Trajectory da imitation) para formatos úteis
    traj_list_dicts, traj_list_objs = trajs_from_imitation_trajectories(expert)

    # Avaliar retornos das demos (opcional)
    mean_ret, std_ret = returns_from_demos(traj_list_objs)
    print(f"Expert demos mean return: {mean_ret} +/- {std_ret}")

    if args.algorithm == "BC":
        # BC espera demonstrações flattenadas (lista de Trajectory objects ou flatten format)
        rollouts_for_bc = rollout_utils.flatten_trajectories(traj_list_objs)

        n_transitions = sum(len(t.obs) for t in traj_list_objs) 
        print(f"[INFO] Número total de transições nas demos: {n_transitions}") 
        # escolher demo_batch_size adaptativo (pelo menos 1) 
        default_batch = 32 
        demo_batch_size = min(default_batch, max(1, n_transitions))

        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=rollouts_for_bc,
            rng=rng,
            batch_size=20,        
        )

        # Avaliar política inicial do BC (pode ser aleatória)
        try:
            reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
            print(f"BC reward before training: {reward_before_training}")
        except Exception as e:
            print(f"[WARN] Não foi possível avaliar política antes do treino: {e}")

        bc_trainer.train(n_epochs=5)

        try:
            reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
            print(f"BC reward after training: {reward_after_training}")
        except Exception as e:
            print(f"[WARN] Não foi possível avaliar política após o treino: {e}")

        try:
            save_policy = PPO(policy=bc_trainer.policy.__class__, env=env, verbose=0)
            save_policy.policy.load_state_dict(bc_trainer.policy.state_dict())
            save_policy.save(args.output)

            print(f"[OK] Política salva em {args.output}")
        except Exception as e:
            print(f"[WARN] Falha ao salvar em formato SB3 (.zip): {e}. State dict salvo em {args.output + '.pt'}")

        env.close()

    elif args.algorithm == "GAIL":
        # GAIL aceita lista de trajectórias (dicts ou Trajectory objects). Usamos os objetos Trajectory.
        rollouts_for_gail = traj_list_objs

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
            demonstrations=rollouts_for_gail,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )

        # Avaliar learner antes do treino adversarial
        try:
            learner_rewards_before_training, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)
        except Exception as e:
            print(f"[WARN] Não foi possível avaliar learner antes do treino: {e}")
            learner_rewards_before_training = []

        num_passos = 200000
        gail_trainer.train(num_passos)
        learner.save(args.output)

        try:
            learner_rewards_after_training, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)
        except Exception as e:
            print(f"[WARN] Não foi possível avaliar learner após o treino: {e}")
            learner_rewards_after_training = []

        if len(learner_rewards_before_training) > 0:
            print(
                "Rewards before training (media):", np.mean(learner_rewards_before_training),
                "com uma derivação +/-", np.std(learner_rewards_before_training),
            )

        if len(learner_rewards_after_training) > 0:
            print(
                "Rewards after training (media):", np.mean(learner_rewards_after_training),
                "com uma derivação+/-", np.std(learner_rewards_after_training),
            )

        env.close()


if __name__ == "__main__":
    main()
