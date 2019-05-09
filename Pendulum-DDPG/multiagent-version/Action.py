from ddpg import Actor
import util
import numpy as np
from normalized_env import NormalizedEnv
import gym
import torch
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim 
        self.mu = mu 
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) *  self.mu

    def reset(self):
        self.X  = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X



class Action():
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor.eval()
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim)
        self.to_tensor = util.to_tensor 
        pass

    def chose_action(self, state, explort):
        if explort:
            a0 = self.get_exploration_action(state)
        else:
            a0 = self.get_exploitation_action(state)
        return a0

    def get_exploitation_action(self, state):
        '''得到给定状态下依据目标演员网络计算出的行为，不探索
            Args:
                state numpy数组
            Returns:
                action numpy数组
        '''
        action = self.actor.forward(self.to_tensor(state)).squeeze(0)
        action = action.cpu().data.numpy()
        return action

    def get_exploration_action(self, state):
        '''得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
            Args：
                state  numpy数组
            Returns:
                action numpy数组
        '''
        action = self.actor.forward(self.to_tensor(state)).squeeze(0)
        new_action = action.cpu().data.numpy() + (self.noise.sample())
        new_action = new_action.clip(min = -1, max = 1)
        return new_action
    
    def load_param(self, source_model):
        self.actor.load_state_dict(source_model.state_dict())



def collect_porcess(agent_index, queue_mem, acrot_param):
    env = NormalizedEnv(gym.make('Pendulum-v0'))
    agent = Action(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    try:
        while True:
            done = False
            state = env.reset()
            state = (state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
            agent.load_param(acrot_param)
            print("agent {} load param".format(agent_index))

            while not done:
                action = agent.chose_action(state, explort=True)
                next_state, reward, done, _ = env.step(action)
                # env.render()
                next_state = (next_state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
                is_done = 0 if done else 1
                queue_mem.put((state, action, next_state, reward, is_done))
                state = next_state
    except Exception as e:
        print(e)
        print("agent {} exit".format(agent_index))
        env.close()

def test_process(config, steps, target_actor):
    env = NormalizedEnv(gym.make('Pendulum-v0'))
    agent = Action(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    reward_list = []
    try:
        while True:
            # for test
            if (steps.value)!=0 and (steps.value % config.test_every_eposide == 0):
                agent.load_param(target_actor)
                print("test agent load param ")
                et_reward = 0
                for index in range(config.num_eposide_test):
                    eposide = 0
                    state = env.reset()
                    state = (state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)

                    while True:
                        action = agent.chose_action(state, explort=False)
                        next_state, reward, done, _ = env.step(action)
                        env.render()
                        next_state = (next_state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
                        eposide += reward
                        state = next_state
                        if done:
                            break
                    et_reward += eposide    
                print("\033[93m [ test ] eposide average reward : {}\033[00m" .format(et_reward/config.num_eposide_test))
                reward_list.append(et_reward/config.num_eposide_test)
                        
                x = np.arange(len(reward_list))
                y = np.array(reward_list)
                plt.plot(x, y)
                plt.savefig("./eposide_reward.png")

                
    except Exception as e:
        print(e)
        print("test process exit")
        env.close()

