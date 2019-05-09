import torch
import torch.nn  as nn 
import torch.nn.functional as F
from model import Actor
from model import Critic
import util

import numpy as np
import gym

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

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.batch_size = batch_size
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.weight_decay= weight_decay
        self.gamma = gamma
        self.tau = 0.001

        self._to_tensor = util.to_tensor
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters(),self.learning_rate_actor, weight_decay=self.weight_decay)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate_critic, weight_decay=self.weight_decay)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.t = 0


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
        s1 = self._to_tensor(s1, device=self.device)
        s0 = self._to_tensor(s0, device=self.device)

        next_q_values = self.target_critic.forward(
            state=s1,
            action=self.target_actor.forward(s1)
        ).detach()
        target_q_batch = self._to_tensor(r1, device=self.device) + \
            self.gamma*self._to_tensor(terminal_batch.astype(np.float), device=self.device)*next_q_values
        q_batch = self.critic.forward(s0, self._to_tensor(a0, device=self.device))

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


    def learning(self, memory):
        self.actor.train()
        return self._learn_from_memory(memory)

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('Models loaded successfully')


