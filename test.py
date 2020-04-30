import core
import gym
import time
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pre_processing = transforms.Compose([
        transforms.Lambda(lambda x: x[:195,:,:]),
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.numpy()),
        # transforms.ToPILImage(),

    ])


env = gym.make('PongDeterministic-v4')
o = env.reset()
print(o.shape)
o = pre_processing(o)
print(o.shape)
plt.imshow(o.squeeze(0))
# plt.ion()
plt.show()
# plt.pause(0.5)

o = pre_processing(o)
memory = [np.copy(o) for _ in range(4)]
model = core.CNNActorCritic(env.observation_space, env.action_space).to(device)
# model.load_state_dict(torch.load('model.pth'))

while True:
    o = np.concatenate(memory)
    a, v, logp = model.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device))

    next_o, r, d, _ = env.step(a)
    plt.imshow(next_o)
    plt.ion()
    plt.show()
    plt.pause(0.2)
    env.render()
    time.sleep(0.2)
    o = pre_processing(next_o)
    memory.pop(0)
    memory.append(o)

    if d:
        break

