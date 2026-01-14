import argparse
import pickle
import gymnasium as gym
import seals
import numpy as np
from stable_baselines3 import PPO
from imitation.data import types

# Importar o ambiente Custom
from custom import Custom

# Registar o ambiente Custom no gymnasium
gym.register(
    id='Custom-v0',
    entry_point='custom:Custom',
    kwargs={'n': 10, 'm': 10, 'num_k': 15, 'max_steps': 100}
)


def get_action_from_keyboard(env_name):

    if env_name == "CartPole":
        print("\nControlos: [a] Esquerda | [d] Direita | [q] Sair")
        key = input("Ação: ").strip().lower()

        if key == 'q':
            return None
        elif key == 'a':
            return 0
        elif key == 'd':
            return 1
        else:
            print("Tecla inválida! Usa 'a', 'd' ou 'q'")
            return get_action_from_keyboard(env_name)

    else:  # Custom
        print("\nControlos: [w] Cima | [s] Baixo | [a] Esquerda | [d] Direita | [q] Sair")
        key = input("Ação: ").strip().lower()

        if key == 'q':
            return None
        elif key == 'w':
            return 0
        elif key == 's':
            return 1
        elif key == 'a':
            return 2
        elif key == 'd':
            return 3
        else:
            print("Tecla inválida! Usa 'w', 'a', 's', 'd' ou 'q'")
            return get_action_from_keyboard(env_name)


def collect_demonstrations_manual(env, env_name, num_episodes):
    """
    Coleta demonstrações através de controlo manual do utilizador
    Retorna uma lista de Trajectories compatível com imitation
    """
    trajectories = []

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"EPISÓDIO {ep + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Listas para armazenar dados do episódio
        obs_list = []
        acts_list = []
        infos_list = []

        # Reset do ambiente
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Visualização inicial
        if env_name == "Custom":
            env.unwrapped.draw_env()

        terminated = False
        truncated = False
        step_count = 0

        print(f"\nObservação inicial: {obs}")

        while not (terminated or truncated):
            # Obter ação do utilizador
            action = get_action_from_keyboard(env_name)

            if action is None:
                print("\nEpisódio cancelado pelo utilizador.")
                break

            # Guardar observação e ação
            obs_list.append(obs)
            acts_list.append(action)
            infos_list.append(info)

            # Executar ação
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False

            step_count += 1

            # Visualização
            print(f"\n--- Passo {step_count} ---")
            if env_name == "Custom":
                env.unwrapped.draw_env()

            print(f"Observação: {obs}")
            print(f"Reward: {reward:.2f}")

            if terminated:
                print("Objetivo alcançado!")
            elif truncated:
                print("⏱Tempo esgotado!")

        # Se o episódio foi completado (não cancelado)
        if action is not None:
            # Adicionar observação final
            obs_list.append(obs)

            # Criar Trajectory para este episódio
            # obs tem N+1 elementos (inicial + após cada ação)
            # acts e infos têm N elementos (um para cada ação)
            trajectory = types.Trajectory(
                obs=np.array(obs_list),
                acts=np.array(acts_list),
                infos=np.array(infos_list) if infos_list else None,
                terminal=True
            )
            trajectories.append(trajectory)

            print(f"\nEpisódio {ep + 1} concluído!")
            print(f"   Passos: {step_count}")
            print(f"   Terminated: {terminated}, Truncated: {truncated}")
        else:
            print(f"\nEpisódio {ep + 1} foi cancelado.")
            resposta = input("Repetir este episódio? (s/n): ").strip().lower()
            if resposta == 's':
                ep -= 1  # Repetir este episódio

    return trajectories


def collect_demonstrations_ppo(env, num_episodes, model_path="ppo-CartPole-v1"):
    """
    Coleta demonstrações usando uma policy PPO pré-treinada
    Retorna uma lista de Trajectories compatível com imitation
    """
    print("\n🤖 Modo automático com PPO")

    try:
        # Tentar carregar modelo local
        model = PPO.load(model_path)
        print(f"Modelo PPO carregado de {model_path}")
    except:
        print(f"Não foi possível carregar {model_path}")
        print("A tentar descarregar modelo do HuggingFace...")
        try:
            # Alternativa: descarregar do HuggingFace
            from huggingface_sb3 import load_from_hub
            checkpoint = load_from_hub(
                repo_id="sb3/ppo-CartPole-v1",
                filename="ppo-CartPole-v1.zip",
            )
            model = PPO.load(checkpoint)
            print("Modelo descarregado do HuggingFace com sucesso!")
        except Exception as e:
            print(f"ERRO: Não foi possível carregar modelo PPO: {e}")
            print("Por favor, treina um modelo PPO primeiro ou usa controlo manual.")
            return []

    trajectories = []

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"EPISÓDIO {ep + 1}/{num_episodes} (PPO automático)")
        print(f"{'=' * 60}")

        obs_list = []
        acts_list = []
        infos_list = []

        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated):
            # Policy PPO escolhe a ação
            action, _ = model.predict(obs, deterministic=True)

            # Guardar dados
            obs_list.append(obs)
            acts_list.append(action)
            infos_list.append(info)

            # Executar ação
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False

            step_count += 1

        # Adicionar observação final (mas NÃO info extra)
        obs_list.append(obs)

        # Criar Trajectory
        trajectory = types.Trajectory(
            obs=np.array(obs_list),
            acts=np.array(acts_list),
            infos=np.array(infos_list) if infos_list else None,
            terminal=True
        )
        trajectories.append(trajectory)

        print(f"Episódio {ep + 1} concluído com {step_count} passos")

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Gerar demonstrações de um ginásio")
    parser.add_argument("--gym", type=str, required=True,
                        choices=["CartPole", "Custom"],
                        help="Nome do ginásio")
    parser.add_argument("--episodes", type=int, required=True,
                        help="Número de episódios de demonstração")
    parser.add_argument("--file", type=str, required=True,
                        help="Ficheiro de output para guardar demonstrações")
    parser.add_argument("--use_ppo_huggingface", action="store_true",
                        help="Usar policy PPO pré-treinada (apenas CartPole)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GERADOR DE DEMONSTRAÇÕES")
    print("=" * 60)

    # Criar ambiente
    if args.gym == "CartPole":
        env = gym.make("CartPole-v1")  # Usar CartPole padrão que termina corretamente
        env_name = "CartPole"
    else:
        env = gym.make("Custom-v0")
        env_name = "Custom"

    print(f"\nConfiguração:")
    print(f"   Ambiente: {env_name}")
    print(f"   Episódios: {args.episodes}")
    print(f"   Ficheiro de saída: {args.file}")

    # Coletar demonstrações
    if args.use_ppo_huggingface and args.gym == "CartPole":
        trajectories = collect_demonstrations_ppo(env, args.episodes)
    else:
        print(f"\nModo: Controlo manual")
        trajectories = collect_demonstrations_manual(env, env_name, args.episodes)

    if len(trajectories) == 0:
        print("\nNenhuma demonstração foi coletada. A sair...")
        env.close()
        return

    # Guardar demonstrações no formato correto
    try:
        with open(args.file, "wb") as f:
            pickle.dump(trajectories, f)

        print(f"\n{'=' * 60}")
        print(f"SUCESSO!")
        print(f"{'=' * 60}")
        print(f"{len(trajectories)} demonstrações guardadas em: {args.file}")

        # Estatísticas
        total_steps = sum(len(traj.acts) for traj in trajectories)
        avg_steps = total_steps / len(trajectories) if trajectories else 0

        print(f"\nEstatísticas:")
        print(f"   Total de passos: {total_steps}")
        print(f"   Média de passos por episódio: {avg_steps:.1f}")
        print(f"   Min passos: {min(len(traj.acts) for traj in trajectories)}")
        print(f"   Max passos: {max(len(traj.acts) for traj in trajectories)}")
        print(f"\n{'=' * 60}")

    except Exception as e:
        print(f"\nERRO ao guardar ficheiro: {e}")

    env.close()


if __name__ == "__main__":
    main()