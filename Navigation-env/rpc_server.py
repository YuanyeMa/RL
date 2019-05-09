import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
import math, sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages/')
sys.path.append('/usr/local/lib64/python2.7/site-packages/')
from playerc import *
import numpy  as np


class Robot:
    def __init__(self, port):
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

        self.sp = playerc_simulation(self.robot, 0)
        if self.sp.subscribe(PLAYERC_OPEN_MODE):
            raise Exception(playerc_error_str())
    
    def close(self):
        self.laser.unsubscribe()
        self.rp.unsubscribe()
        self.tp.unsubscribe()
        self.sp.unsubscribe()
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
        robot_name = 'r0'
        return self.sp.set_pose2d(robot_name, ppx, ppy, ppa)

    def get_target_pose(self):
        self.robot.read()
        return [self.tp.px, self.tp.py]

    def set_target_pose(self, tx, ty):
        target_name = 't0'
        return self.sp.set_pose2d(target_name, tx, ty, 0)

    def set_robot_cmd_vel(self, vx, vy, va):
        return self.rp.set_cmd_vel(vx, vy, va, 1)

class KeyValueServer:
    _rpc_methods_ = ['get_robot_state', 'get_laser_data', 'set_robot_pose', 'set_target_pose', 'set_robot_cmd_vel']

    def __init__(self, server_port, robot_port):
        self.robot = Robot(robot_port)
        self._serv = SimpleXMLRPCServer(("", server_port), allow_none=True)
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

def rpc_server(server_port, robot_port):
    print("\033[92m rpc server pid : {} rpc server on port : {}\033[00m".format(os.getpid(), server_port))
    kvserv = KeyValueServer(server_port, robot_port)
    try:
        kvserv.serve_forever()
    except Exception as e:
        print(e)
    finally:
        print("close connection port : {}".format(server_port))
        kvserv.close()

def main(argv):
    server_port = int(argv[0])
    robot_port = int(argv[1])
    rpc_server(server_port, robot_port)
    pass

if __name__ == '__main__':
    main(sys.argv[1:])



