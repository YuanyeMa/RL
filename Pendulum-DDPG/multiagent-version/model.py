import torch
import torch.nn  as nn 
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
    '''一种较为合理的初始化网络参数的方法，参考：https://arxiv.org/abs/1502.01852'''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return  torch.Tensor(size).uniform_(-v, v) #从 -v到v的均匀分布  


class Critic(nn.Module):
    def __init__ (self, state_dim, action_dim):
        ''' 构建一个评论家模型 
            该网络属于价值函数近似的第二种类型， 根据状态和行为输出一个价值
            Args：
                state_dim: 状态特征的数量
                action_dim: 行为作为输入的特征的数量
        '''
        super(Critic, self).__init__()
     
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400+action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        '''前向运算， 根据状态和行为的特征得到评论家给出的价值
            Args:
                state: 状态的特征表示 torch Tensor [n, state_dim]
                action: 行为的特征表示 torch Tensor [n, action_dim]
            Returns:
                Q(s,a) Torch Tensor [n, 1]
        '''
        s1 = F.relu(self.fc1(state))
        s2 = torch.cat((s1, action), dim=1)
        s3 = F.relu(self.fc2(s2))
        out = self.fc3(s3)
        return out


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        ''' 构建一个演员模型
            Args:
                state_dim: 状态特征的数量
                action_dim: 行为作为输入的特征的数量
        '''
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-EPS, EPS)


    def forward(self, state):
        '''前向运算，根据状态的特征表示得到具体的行为值
            Args:
                state: 状态的特征表示 torch Tensor [n, state_dim]
            Rerurns:
                action: 行为的特征表示 torch Tensor [n, action_dim]
        '''
        out = self.fc1(state)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        action = torch.tanh(out)

        return action 
