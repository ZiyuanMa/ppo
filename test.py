import core
import gym
import time
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pre_processing = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())

    ])


env = gym.make('MsPacman-v0')
o = env.reset()
# plt.imshow(o)
# plt.ion()
# plt.show()
# plt.pause(0.5)

o = pre_processing(o)
model = core.CNNActorCritic(env.observation_space, env.action_space).to(device)
model.load_state_dict(torch.load('model.pth'))

while True:
    
    a, v, logp = model.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device))

    next_o, r, d, _ = env.step(a)
    # plt.imshow(next_o)
    # plt.ion()
    # plt.show()
    # plt.pause(0.2)
    env.render()
    time.sleep(0.2)
    o = pre_processing(next_o)

    if d:
        break

