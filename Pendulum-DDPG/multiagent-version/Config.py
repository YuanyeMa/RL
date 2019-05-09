import numpy as np

class Config:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.capacity = 1e6
        self.max_episode_num = 2001
        self.batch_size = 64
        self.save_steps = 10000
        #self.devices 

        self.seed = 2
        self.plot = False
        self.weight_decay = 0.01
        self.learning_rate_actor = 10e-3
        self.learning_rate_critic = 10e-4
        self.learning_steps = 20
        self.warmup = 100
        self.test_every_eposide = 20
        self.num_eposide_test = 10
        self.agent_num = 5
     