from collections import namedtuple
import random

import gym

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'is_done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    @property
    def len(self):
        return len(self.memory)

if __name__ == '__main__':
    memory = ReplayMemory(capacity=10)

    env = gym.make('Pendulum-v0')
    s = env.reset()
    i = 0

    while i<10:
        action = env.action_space.sample()
        s_, r, done, _ = env.step(action)
        env.render()
        memory.push(s, action, s_, r)
        s = s_
        i+=1
    
    env.close()

    transitions = memory.sample(2)
    batch = Transition(*zip(*transitions))
    print(batch)
    print(batch.state[0])
    print(memory.len)
