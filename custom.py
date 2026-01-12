import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Custom(gym.Env):

    def __init__(self, n=10, m=10, num_k=15, max_steps=100):
        super(Custom, self).__init__()
        
        self.n = n # tamanho da grid nxm
        self.m = m # tamanho da grid nxm
        self.num_k = num_k # num de paredes
        self.max_steps = max_steps

        self.pos_A = None # posicao do agente
        self.pos_G = None # posicao do goal
        self.pos_k = set() # posicao das paredes
        self.step_count = 0
        
        # acoes: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.action_space = spaces.Discrete(4)

        # Observation Space 
        # The agent observes:
        # 1. Own position (row, col)
        # 2. Presence of walls/obstacles in contiguous cells (Up, Down, Left, Right) - Binary (0 or 1)
        # 3. Relative position of the goal (d_row, d_col)
        # Total vector size: 2 + 4 + 2 = 8
        
        # Define bounds for observations to ensure compatibility with Stable Baselines3/Imitation
        # Min values: [0, 0, 0, 0, 0, 0, -n, -m]
        # Max values: [n, m, 1, 1, 1, 1, n, m]
        low = np.array([0, 0, 0, 0, 0, 0, -self.n, -self.m], dtype=np.int32)
        high = np.array([self.n, self.m, 1, 1, 1, 1, self.n, self.m], dtype=np.int32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.pos_k = set()

        # posicao random do goal
        self.pos_G = (
            self.np_random.integers(0, self.n),
            self.np_random.integers(0, self.m)
        )

        # posicao random do agente
        while True:
            self.pos_A = (
                self.np_random.integers(0, self.n),
                self.np_random.integers(0, self.m)
            )
            if self.pos_A != self.pos_G:
                break
        
        # posicoes random das paredes
        newpos_k = []
        while len(newpos_k)<self.num_k:
            cords = (
                self.np_random.integers(0, self.n),
                self.np_random.integers(0, self.m)
            )
            if (cords != self.pos_A) and (cords != self.pos_G) and (cords not in newpos_k):
                newpos_k.append(cords)
                self.pos_k.add(cords)
                # print(cords)
            

        observation = self._get_obs()
        
        return observation

    def _get_obs(self):
        n, m = self.pos_A
        
        # ve se existem pareces em cada acao possivel
        w_up = 1 if (n-1 < 0) or ((n-1, m) in self.pos_k) else 0
        w_down = 1 if (n+1 >= self.n) or ((n+1, m) in self.pos_k) else 0
        w_left = 1 if (m-1 < 0) or ((n, m-1) in self.pos_k) else 0
        w_right = 1 if (m+1 >= self.m) or ((n, m+1) in self.pos_k) else 0

        dn = self.pos_G[0] - n
        dm = self.pos_G[1] - m

        return np.array([n, m, w_up, w_down, w_left, w_right, dn, dm], dtype=np.int32)

    def step(self, action):
        self.step_count += 1
        
        # acoes disponiveis
        next_move = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        
        target_n = self.pos_A[0] + next_move[action][0]
        target_m = self.pos_A[1] + next_move[action][1]
        next_pos = (target_n, target_m)

        # ver se tem parede/out of bounds
        in_bounds = (0 <= target_n < self.n) and (0 <= target_m < self.m)
        in_wall = next_pos in self.pos_k
        
        if in_bounds and not in_wall:
            self.pos_A = next_pos

        terminated = (self.pos_A == self.pos_G)
        truncated = (self.step_count >= self.max_steps)

        reward = -1 
        if not in_bounds or in_wall:
            reward -= 1 # penalizacao de parede/out of bounds
        if terminated:
            reward += 10 # reward do objetivo

        observation = self._get_obs()

        return observation, reward, terminated, truncated

    def draw_env(self):
        print(f"\nStep {self.step_count}")
        for n in range(self.n):
            row_str = ""
            for m in range(self.m):
                pos = (n, m)
                if pos == self.pos_A:
                    row_str += "A " # agente
                elif pos == self.pos_G:
                    row_str += "G " # goal
                elif pos in self.pos_k:
                    row_str += "# " # parede
                else:
                    row_str += "_ "
            print(row_str)
        print("\n")

    def close(self):
        pass

if __name__ == "__main__":
    env = Custom(n=6, m=6, num_k=6)
    obs = env.reset()
    env.draw_env()
    print(obs)

    for i in range(2):
        action = env.action_space.sample()
        obs, reward, terminated, truncated = env.step(action)
        env.draw_env()
        print(obs)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")