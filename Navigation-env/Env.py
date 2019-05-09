"""
"""
import os
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import namedtuple

from Robot import Robot
from Server import Server

RAD2DEG = 57.29577951308232     # 弧度与角度换算关系1弧度=57.29..角度

class RobotWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
        }

    server_port = 16000
    robot_port = 8000

    @classmethod
    def _get_server_port(cls):
        return cls.server_port

    @classmethod
    def _get_robot_port(cls):
        return cls.robot_port

    def __init__(self, index):
        print("\033[92m ======================================= \033[00m")
        self.index = index
        # 1. 初始化server
        try:
            self.server = Server(server_port=RobotWorld._get_server_port()+self.index,
                             robot_port=RobotWorld._get_robot_port()+self.index)
            self.server.start_server()
            # 2. 初始化robot (client)
            self.server_address = 'http://localhost:'+str(RobotWorld._get_server_port()+self.index)
            self.robot = Robot(self.server_address, self.index)
        except Exception as e:
            print("\033[31m ERROR : {} \033[00m".format(e))
            raise Exception("Create Robot object failed") 
        
        print("\033[92m robot world {} create succeccful \033[00m".format(self.index))
        
        self.r_arrive = 25  # 到达目标点的奖励
        self.r_wg = 10 # 距离缩短时的奖励系数
        self.r_collision = -15 # 碰撞时的惩罚
        self.dis = 0 # 上一个距离目标的距离

        self.scenes_low_x = -8            
        self.scenes_high_x = 8
        self.scenes_low_y = -8            
        self.scenes_high_y = 8

        self.max_line_speed = 1     # max agent velocity along a axis
        self.max_angle_speed = np.pi/2
        
        # action是2维的，[0]表示x线速度， [1]表示转向，角度用弧度表示，[-np.pi/2,  np.pi/2]
        self.low_action = np.array([0, -self.max_angle_speed]) 
        self.high_action = np.array([self.max_line_speed,  self.max_angle_speed])

        self.rad = 0.5             # agent 半径,目标半径
        self.target_rad = 0.3      # target radius.
        self.goal_dis = 0.8       # 目标接近距离 expected goal distance
    
        # 作为观察空间每一个特征值的下限
        self.low_state = np.concatenate((np.zeros(180, dtype=float),    # laser data
                                        np.array([-8.0, -8.0, -np.pi]), # robot pose
                                        np.array(self.low_action),      # robot action now
                                        np.array([-7.0, -7.0])          # target pose
                                        ), axis=0) 

        self.high_state = np.concatenate((np.array([4]*180),            # laser data
                                        np.array([8.0, 8.0, np.pi]),    # robot pose
                                        np.array( self.high_action),    # robot action now
                                        np.array([7.0, 7.0])            # target pose
                                        ), axis=0) 


        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        self.observation_space = spaces.Box(self.low_state, self.high_state)    

        self.state = None
        self.seed()
        self.t = 0
        



    def step(self, action):
        self.t += 1
        # 1. 执行动作
        #print("-- from env step action:{} ".format(action))
        self.robot.set_robot_cmd_vel(float(action[0]), 0, float(action[1]))

        # 2. 获取执行后的状态
        if self.state is None:
            self.state = self.reset()
        else:
            self.state = self.get_state()
        tx = self.state[185]
        ty = self.state[186]
        px = self.state[180]
        py = self.state[181]

        dx = px-tx
        dy = py-ty
        dis = self._compute_dis(dx, dy)

        done = False
        # 3. 计算reward
        if dis <= self.goal_dis:
            self.reward = self.r_arrive
            print('\033[97m Agent {} Yahoo !! i get is!! \033[00m'.format(self.index))
            done = True
        elif min(self.state[0:180])<=self.rad:
            self.reward = self.r_collision
            print('Agent {} Duang!!! '.format(self.index))
            done = True
        else :
            self.reward = self.r_wg*(self.dis-dis)
        self.dis = dis
        return self.state, self.reward, done, {}


    def reset(self):
        self.target_pose = self._random_pos()
        self.robot.set_target_pose(float(self.target_pose[0]), float(self.target_pose[1]))
        px, py = self._random_pos()
        pa = self.np_random.uniform(low = -np.pi, high = np.pi)
        self.robot.set_robot_pose(float(px), float(py), float(pa))

        dx = px - self.target_pose[0]
        dy = py - self.target_pose[1]
        self.state = self.get_state()
        self.dis = self._compute_dis(dx, dy)     # 上一个距离目标的距离
        return self.state   

    def get_state(self):
        laser_data = np.zeros(180, dtype=float)
        laser_data = self.robot.get_laser_data()

        (ppx, ppy, ppa, vx, vy, va) = self.robot.get_robot_state()

        robot_pose = np.zeros(3, dtype=np.float)
        robot_pose[0] = ppx
        robot_pose[1] = ppy
        robot_pose[2] = ppa

        robot_action = np.zeros(2, dtype=np.float)
        robot_action[0] = vx
        robot_action[1] = va
        
        state = np.concatenate((laser_data, robot_pose, robot_action, self.target_pose), axis=0)
        return state

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)  
        return [seed]

    def _clip(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        return x

    def _random_pos(self):
        target_pose = np.zeros(2, dtype=np.float)
        target_pose[0] = self.np_random.uniform(low = self.scenes_low_x+1, high = self.scenes_high_x-1)
        target_pose[1] = self.np_random.uniform(low = self.scenes_low_y+1, high = self.scenes_high_y-1)
        return target_pose

    def _compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx,2) + math.pow(dy,2))

    def close(self):
        if hasattr(self, "robot"):
            self.robot.set_robot_cmd_vel(0, 0, 0)
            self.robot.set_robot_pose(0,0,0)
        self.server.stop_server()
        print("Agent {} close".format(self.index))
