from xmlrpc.client import ServerProxy

class Robot:
    def __init__(self, address, index):
        self.index = index
        try:
            self.s = ServerProxy(address, allow_none=True)
        except Exception as e:
            print("\033[31m ERROR : {} \033[00m".format(e))
            raise Exception("connect to {} failed".format(address)) 

    def get_robot_state(self):
        return self.s.get_robot_state()

    def get_laser_data(self):
        return self.s.get_laser_data()

    def set_robot_pose(self, ppx, ppy, ppa):
        return self.s.set_robot_pose(ppx, ppy, ppa)

    def set_target_pose(self, tpx, tpy):
        return self.s.set_target_pose(tpx, tpy)

    def set_robot_cmd_vel(self, vx, vy, va):
        return self.s.set_robot_cmd_vel(vx, vy, va)
    
    def close(self):
        pass

