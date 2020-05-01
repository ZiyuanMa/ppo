import gym
import numpy as np
from torchvision import transforms
from typing import Dict, List, Optional

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

    def reset(self, env_ids: List[int]):


        for env_id in env_ids:
            o = self.envs[env_id].reset()
            o = pre_processing(o)
            self.memory[env_id] = [np.copy(o) for _ in range(4)]


        return [np.concatenate(memory) for memory in self.memory]

    def step(self, actions):
        rews = []
        done = []

        assert len(actions) == len(self.envs), '{} actions but {} environments'.format(len(actions), len(self.envs))

        for env, action, memory in zip(self.envs, actions, self.memory):
            o, r, d, _ = env.step(action)
            o = pre_processing(o)
            memory.pop(0)
            memory.append(o)
            rews.append(r)
            done.append(d)

        return [np.concatenate(memory) for memory in self.memory], rews, done


        