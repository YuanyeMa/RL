import os, signal
import multiprocessing as mp
import subprocess
import time 
import psutil


class Server:
    def __init__(self, robot_port=7005, server_port=16000):
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.robot_port = robot_port
        self.server_port = server_port
        self.player_pid = None 
        self.server_pid = None
        pass

    def start_server(self):
        WORLD_FILE = self.path+"/world/simple.cfg"
        cmd = "player "+WORLD_FILE+" -p "+str(self.robot_port)
        print("\033[92m CMD : {} \033[00m".format(cmd))
        self._process_player = subprocess.Popen(cmd, shell=True)
        time.sleep(2)

        pid = subprocess.getoutput("lsof -i :"+str(self.robot_port)+"|  grep player | awk '{print $2}'").split('\n')[0]
        if pid is not '':
            self.player_pid = int(pid)
            print("\033[92m player server pid : {} \033[00m".format(self.player_pid))

        RPC_SERVER_PATH = self.path+"/rpc_server.py "
        cmd = "python2 "+RPC_SERVER_PATH+str(self.server_port)+" "+str(self.robot_port)+"  2>/dev/null"
        print("\033[92m CMD : {} \033[00m".format(cmd))
        self._process_server = subprocess.Popen(cmd, shell=True)
        time.sleep(2)
        pid = subprocess.getoutput("lsof -i :"+str(self.server_port)+"| grep python2 | awk '{print $2}'")
        if pid is not '':
            self.server_pid = int(pid)
            print("\033[92m rpc server pid : {} \033[00m".format(self.server_pid))
        
        assert self.player_pid is not None and self.server_pid is not None

        
    def is_player_alive(self):
        p = psutil.Process(self.player_pid)
        return p.is_running()
        
    def is_rpc_alive(self):
        p = psutil.Process(self.server_pid)
        return p.is_running()
        
    def stop_server(self):
        if self.server_pid is not None:
            try:
                result = os.kill(self.server_pid, signal.SIGKILL)
                print(" Kill rpc_server {} : {}".format(self.server_pid, result))
                self.server_pid = None
            except OSError:
                print(" No {}".format(self.server_pid))

        if self.player_pid is not None  :
            try:
                result = os.kill(self.player_pid, signal.SIGKILL)
                print(" Kill player {} : {}".format(self.player_pid, result))
                self.player_pid = None
            except OSError:
                print(" No {}".format(self.player_pid))

    def __del__(self):
        if self.player_pid is not None or self.server_pid is not None:
            self.stop_server()
        