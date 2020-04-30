import gym
import numpy as np
from torchvision import transforms

pre_processing = transforms.Compose([
        transforms.Lambda(lambda x: x[:195,:,:]),
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy()),

    ])

class VecEnv:
    def __init__(self, name, num):
        self.envs = [gym.make(name) for _ in range(num)]
        self.memory = [[] for _ in range(num)]

    def reset(self):
        obs = [pre_processing(env.reset()) for env in self.envs]
        for memory, o in zip(self.memory, obs):
            memory.clear()
            memory = [np.copy(o) for _ in range(4)]

        return obs

    def step(self, actions):
        rets = []

        assert len(actions) == len(self.envs), '{} actions but {} environments'.format(len(actions), len(self.envs))

        for env, action, memory in zip(self.envs, actions, self.memory):
            o, r, d, _ = env.step(action)
            o = pre_processing(o)
            memory.pop(0)
            memory.append(o)
            rets.append((np.concatenate(memory), r, d))

        return rets


        