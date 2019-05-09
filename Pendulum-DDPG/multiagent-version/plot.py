import numpy as np 
import matplotlib.pyplot as plt
import time
from visdom import Visdom  
import random


class Vis:
    def __init__(self, env='default'):
        self.vis = Visdom(env=env)
        self.lines = {}
        self.win_action_bar = None


    def update(self, x, y, line_name, **kwargs):
        X = np.array([x])
        Y = np.array([y])
        if line_name not in self.lines.keys():
            self.lines[line_name] = self.vis.line(X=X, Y=Y, opts=dict(title=line_name, **kwargs))
        self.vis.line(X=X, Y=Y, win=self.lines[line_name],update='append', opts=dict(title=line_name, **kwargs))

    def show_action(self, action):
        if self.win_action_bar is None:
            self.win_action_bar = self.vis.bar(X=action, opts=dict(rawnames=['line_vel', 'angle_vel']))

        self.win_action_bar = self.vis.bar(X=action, win=self.win_action_bar, opts=dict(rawnames=['line_vel', 'angle_vel']))
        



def main():
    vis=Vis(env='main')
    for i in range(10):
        y1 = random.uniform(-10, 10)
        y2 = random.uniform(-10, 10)
        x=np.array([i, i])
        y=np.array([y1, y2])
        vis.update(x, y, line_name='critic_q_value', showlegend=True, legend=['aaaaa', 'bbbbbb'])
        print(x, y)


if __name__ == '__main__':
    main()


