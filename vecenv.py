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
        self.histories = [[] for _ in range(num)]
        self.done_env_ids = []

    def reset(self):
        if len(self.done_env_ids) == 0:
            # reset all environments
            for env_id, env in enumerate(self.envs):
                o = env.reset()
                o = pre_processing(o)
                self.histories[env_id] = [np.copy(o) for _ in range(4)]

        else:
            # reset done environments
            for env_id in self.done_env_ids:
                o = self.envs[env_id].reset()
                o = pre_processing(o)
                self.histories[env_id] = [np.copy(o) for _ in range(4)]

            self.done_env_ids.clear()
            


        return [np.concatenate(history) for history in self.histories]

    def step(self, actions):
        rews = []
        dones = []

        assert len(self.done_env_ids) == 0, 'need reset env'
        assert len(actions) == len(self.envs), '{} actions but {} environments'.format(len(actions), len(self.envs))

        for env, action, history in zip(self.envs, actions, self.histories):
            o, r, d, _ = env.step(action)
            o = pre_processing(o)

            history.pop(0)
            history.append(o)

            # reward clipping
            if r > 10:
                r = 10
            elif r < -10:
                r = -10

            rews.append(r)
            dones.append(d)

        return [np.concatenate(history) for history in self.histories], rews, dones


        