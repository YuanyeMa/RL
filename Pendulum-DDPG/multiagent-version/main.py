import gym
from normalized_env import NormalizedEnv
from ddpg import DDPGAgent
from Action import collect_porcess
from Action import test_process
from Experience import ReplayMemory
from Config import Config 
import numpy as np

from torch.backends import cudnn
#import multiprocessing as mp
import torch.multiprocessing as mp

def main():
    mp.set_start_method('spawn')    
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
    agent.target_actor.share_memory()
    # 3. 初始化memory
    memory = ReplayMemory(config.capacity)

    q = mp.Queue(10)

    process_collect_list = []
    for i in range(config.agent_num):
        process_name = "collect_process_"+str(i)
        process = mp.Process(name=process_name, target=collect_porcess, args=(i, q, agent.target_actor))
        process.start()
        process_collect_list.append(process)

    steps = mp.Value('d', 0)
    test_p = mp.Process(name="test_process", target=test_process, args=(config, steps, agent.target_actor))
    test_p.start()
    process_collect_list.append(test_p)
    
    try:
        while True:
            len = q.qsize()
            while len:
                mem = q.get()
                memory.push(mem[0], mem[1], mem[2], mem[3], mem[4])
                len -= 1
            # 4.4 学习
            if memory.len>config.batch_size:
                agent.learning(memory)
            # save model
            if steps.value>1 and steps.value%config.save_steps==0:
                agent.save_models(steps.value/config.save_steps)   
            steps.value += 1  
    except Exception as e:
        print(e)
    except:
        for process in process_collect_list:
            process.join()
            print(process.name+" stop ")
    env.close()

if __name__ == '__main__':
    main()




