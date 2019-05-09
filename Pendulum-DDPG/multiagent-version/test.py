import gym
import torch
from ddpg import Actor
import numpy as np
from normalized_env import NormalizedEnv
from plot import Vis

def to_tensor(ndarray, requires_grad=False, dtype=torch.float):
    return torch.tensor(ndarray, dtype = dtype, device = 'cuda', requires_grad=requires_grad)


def main():
    env = NormalizedEnv(gym.make('Pendulum-v0'))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent =  Actor(state_dim, action_dim).to('cuda')

    agent.load_state_dict(torch.load('./Models/78.0_actor.pt'))

    eposide = 0
    done = False
    eposide_list = []
    while eposide<100:
        eposide_reward = 0
        state = env.reset()
        state = (state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
        state = to_tensor(state)
        while not done:
            action = agent.forward(state).detach().cpu().data.numpy()
            state_, reward, done, _ = env.step(action)
            state_ = (state_-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
            env.render()
            state = to_tensor(state_)
            eposide_reward += reward

        eposide_list.append(eposide_reward)
        eposide+=1
        done = False
        print("{} : {}".format(eposide, eposide_reward))
    
    import matplotlib.pyplot as plt
    x = np.arange(100)
    y = np.array(eposide_list)
    plt.plot(x, y)
    plt.savefig("./test_eposide_reward.png")



    env.close()


if __name__ == '__main__':
    main()

