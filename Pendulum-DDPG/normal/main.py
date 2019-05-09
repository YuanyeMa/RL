import gym
from ddpg import DDPGAgent
from Experience import ReplayMemory
# from Memory import Experience as ReplayMemory
# from Memory  import Transition

from Config import Config 
from tqdm import tqdm
import pickle
import numpy as np
from normalized_env import NormalizedEnv

from torch.backends import cudnn

def main():
    config = Config()
    # 1. 初始化环境
    env = NormalizedEnv(gym.make('Pendulum-v0'))

    # 2. 初始化agent
    agent = DDPGAgent(env=env, 
                        seed = config.seed,
                        batch_size=config.batch_size,
                        learning_rate_actor = config.learning_rate_actor,
                        learning_rate_critic = config.learning_rate_critic,
                        weight_decay=config.weight_decay)
    # 3. 初始化memory
    memory = ReplayMemory(config.capacity)

    steps = 0
    # for eposide_index in tqdm(range(config.max_episode_num)):
    eposide_list = []
    for eposide_index in range(config.max_episode_num):
        state = np.float64(env.reset())
        state = (state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
        episode_reward = 0

        while True:
            steps += 1
            # 4.1 选择一个动作
            if steps < config.warmup:
                action = np.random.uniform(-1, 1)
            else :
                action = agent.chose_action(state, explore=True)

            # 4.2 执行动作，得到下一时刻的状态以及奖励信息
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()

            # 4.3 存储Transition
            next_state = (next_state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
            terminal = 0 if done else 1

            memory.push(state, action, next_state, reward, terminal)
            state = next_state
            # 4.4 学习
            if memory.len>config.batch_size and steps>config.warmup:
                critic_loss, _= agent.learning(memory)
            if done:
                print('#{}: episode_reward: {} steps: {}'.format(eposide_index, episode_reward, steps))
                eposide_list.append(episode_reward)
                break

        # for test
        if eposide_index!=0 and eposide_index % config.test_every_eposide == 0:
            et_reward = 0
            for index in range(config.num_eposide_test):
                state = env.reset()
                state = (state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)

                while True:
                    action = agent.chose_action(state, explore=False)
                    next_state, reward, done, _ = env.step(action)
                    env.render()
                    next_state = (next_state-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
                    et_reward += reward
                    state = next_state
                    if done:
                        break
            print("\033[93m [ test ] eposide average reward : {}\033[00m" .format(et_reward/config.num_eposide_test))
        
        # save model
        if eposide_index>1 and eposide_index%config.save_steps==0:
                agent.save_models(eposide_index)          
            
    env.close()

    import matplotlib.pyplot as plt
    x = np.arange(config.max_episode_num)
    y = np.array(eposide_list)
    plt.plot(x, y)
    plt.savefig("./eposide_reward.png")


if __name__ == '__main__':
    main()




