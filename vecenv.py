import gym

class VecEnv:
    def __init__(self, name, num):
        self.envs = [gym.make(name) for _ in range(num)]

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return obs

    def step(self, actions):
        NotImplemented