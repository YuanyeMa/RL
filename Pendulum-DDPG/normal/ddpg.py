import torch
import torch.nn  as nn 
import torch.nn.functional as F

import numpy as np
import gym
from plot import Vis

EPS = 0.003

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim 
        self.mu = mu 
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) *  self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


def soft_update(target, source, tau):
    '''
        使用下式将source网络(x)参数软更新至target(y)参数： y = tau*x + (1-tau)*y
        Args:
            target: 目标网络（PyTorch）
            source: 源网络 network （PyTorch）
        Returns:
            None
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class DDPGAgent:
    def __init__(self, plot=True,
                        seed = 1,
                        env:gym.Env = None, 
                        batch_size = 128,
                        learning_rate_actor = 0.001,
                        learning_rate_critic = 0.001,
                        weight_decay=0.01,
                        gamma = 0.999):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.plot = plot
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.batch_size = batch_size
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.weight_decay= weight_decay
        self.gamma = gamma

        self.tau = 0.001
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)


        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters(),self.learning_rate_actor, weight_decay=self.weight_decay)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate_critic, weight_decay=self.weight_decay)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.t = 0

    def get_exploitation_action(self, state):
        '''得到给定状态下依据目标演员网络计算出的行为，不探索
            Args:
                state numpy数组
            Returns:
                action numpy数组
        '''
        action = self.target_actor.forward(self._to_tensor(state)).squeeze(0)
        action = action.cpu().data.numpy()
        return action

    def get_exploration_action(self, state):
        '''得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
            Args：
                state  numpy数组
            Returns:
                action numpy数组
        '''
        action = self.actor.forward(self._to_tensor(state)).squeeze(0)
        new_action = action.cpu().data.numpy() + (self.noise.sample())
        new_action = new_action.clip(min = -1, max = 1)
        return new_action

    def _to_tensor(self, ndarray, requires_grad=False, dtype=torch.float):
        return torch.tensor(ndarray, dtype = dtype, device = device, requires_grad=requires_grad)

    def _learn_from_memory(self, memory):
        ''' 从记忆学习，更新两个网络的参数
        '''
        # 随机获取记忆里的Transition
        trans_pieces = memory.sample(self.batch_size)
        s0 = np.vstack([x.state for x in trans_pieces])
        a0 = np.vstack([x.action for x in trans_pieces])
        r1 = np.vstack([x.reward for x in trans_pieces])
        s1 = np.vstack([x.next_state for x in trans_pieces])
        terminal_batch = np.vstack([x.is_done for x in trans_pieces])

        # 优化评论家网络参数
        s1 = self._to_tensor(s1)
        s0 = self._to_tensor(s0)

        next_q_values = self.target_critic.forward(
            state=s1,
            action=self.target_actor.forward(s1)
        ).detach()
        target_q_batch = self._to_tensor(r1) + \
            self.gamma*self._to_tensor(terminal_batch.astype(np.float))*next_q_values
        q_batch = self.critic.forward(s0, self._to_tensor(a0))

        # 计算critic的loss 更新critic网络参数
        loss_critic = F.mse_loss(q_batch, target_q_batch)
        #self.critic_optimizer.zero_grad()
        self.critic.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # 反向传播，以某状态的价值估计为策略目标函数
        loss_actor = -self.critic.forward(s0, self.actor.forward(s0)) # Q的梯度上升
        loss_actor = loss_actor.mean()
        self.actor.zero_grad()
        #self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # 软更新参数
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        return (loss_critic.item(), loss_actor.item())

    def critic_action(self, s0, a0):
        s0 = torch.tensor(s0).type(torch.FloatTensor).unsqueeze(0)
        a0 = torch.tensor(a0).type(torch.FloatTensor).unsqueeze(0)
        return self.critic.forward(s0, a0).squeeze(0).cpu().data.numpy()


    def learning(self, memory):
        return self._learn_from_memory(memory)


    def chose_action(self, state, explore=True):
        #self.actor.eval()
        if explore :
            a0 = self.get_exploration_action(state)
        else:
            a0 = self.get_exploitation_action(state)
        #self.actor.train()

        return a0

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('Models loaded successfully')


