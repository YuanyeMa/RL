import multiprocessing as mp
import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
import math, sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages/')
sys.path.append('/usr/local/lib64/python2.7/site-packages/')
from playerc import *
import numpy  as np

sys.path.append('./../agent')
from config import Config

config = Config()


class SimulationProxy:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SimulationProxy, cls).__new__(cls)
        return cls.instance

    def __init__(self, port):
        self.robot = playerc_client(None, 'localhost', port)
        if self.robot.connect():
            raise Exception(playerc_error_str())
        self.sp = playerc_simulation(self.robot, 0)
        if self.sp.subscribe(PLAYERC_OPEN_MODE):
            raise Exception(playerc_error_str())
        
    def set_target_pose(self, target_name, x, y, a):
        return self.sp.set_pose2d(target_name, x, y, a)


class Robot:
    def __init__(self, port, index, sp):
        self.index = index
        self.sp = sp
        self.robot = playerc_client(None, 'localhost', port)
        if self.robot.connect():
            raise Exception(playerc_error_str())

        self.laser = playerc_ranger(self.robot, 0) 
        if self.laser.subscribe(PLAYERC_OPEN_MODE):
            raise Exception(playerc_error_str())
            
        self.rp =  playerc_position2d(self.robot, 0) # robot position
        if self.rp.subscribe(PLAYERC_OPEN_MODE):
            raise Exception(playerc_error_str())

        self.tp =  playerc_position2d(self.robot, 1) # target position
        if self.tp.subscribe(PLAYERC_OPEN_MODE):
            raise Exception(playerc_error_str())
    
    def close(self):
        self.laser.unsubscribe()
        self.rp.unsubscribe()
        self.tp.unsubscribe()
        self.robot.disconnect()

    def get_laser_data(self):
        self.robot.read()
        data = []
        for i in range(self.laser.ranges_count):
            data.append(self.laser.ranges[i])
        return data

    def get_robot_state(self):
        self.robot.read()
        return [self.rp.px,
                self.rp.py,
                self.rp.pa,
                self.rp.vx,
                self.rp.vy,
                self.rp.va]

    def set_robot_pose(self, ppx, ppy, ppa):
        robot_name = 'r'+str(self.index)
        return self.sp.set_target_pose(robot_name, ppx, ppy, ppa)

    def get_target_pose(self):
        self.robot.read()
        return [self.tp.px, self.tp.py]

    def set_target_pose(self, tx, ty):
        target_name = 't'+str(self.index)
        return self.sp.set_target_pose(target_name, tx, ty, 0)

    def set_robot_cmd_vel(self, vx, vy, va):
        return self.rp.set_cmd_vel(vx, vy, va, 1)

class KeyValueServer:
    _rpc_methods_ = ['get_robot_state', 'get_laser_data', 'set_robot_pose', 'set_target_pose', 'set_robot_cmd_vel']

    def __init__(self, address, index, sp):
        self.robot = Robot(config.player_port+index, index, sp)
        self._serv = SimpleXMLRPCServer(address, allow_none=True)
        for name in self._rpc_methods_:
            self._serv.register_function(getattr(self, name))

    def get_robot_state(self):
        return self.robot.get_robot_state()

    def get_laser_data(self):
        return self.robot.get_laser_data()

    def set_robot_pose(self, ppx, ppy, ppa):
        return self.robot.set_robot_pose(ppx, ppy, ppa)

    def set_target_pose(self, ppx, ppy):
        return self.robot.set_target_pose(ppx, ppy)

    def set_robot_cmd_vel(self, vx, vy, va):
        return self.robot.set_robot_cmd_vel(vx, vy, va)

    def serve_forever(self):
        self._serv.serve_forever()

    def close(self):
        self._serv.shutdown()
        self._serv.server_close()

def rpc_server(port, index, sp):
    print("------ pid : {} rpc server on port : {}".format(os.getpid(), port))
    kvserv = KeyValueServer(("", port), index, sp)
    try:
        kvserv.serve_forever()
    except:
        print("close connection port : {}".format(port))
        kvserv.close()


if __name__ == '__main__':
    processes = []
    sp = SimulationProxy(config.player_port-1)

    for i in range(1):
        port = config.server_port+i
        process = mp.Process(target=rpc_server, args=(port, i, sp))
        process.start()
        processes.append(process)

    try :
        while True:
            pass
    except:
        for process in processes:
            process.join()
            print("stop {}".format(process.name))




