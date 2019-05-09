import os 
import multiprocessing as mp
import subprocess
import time 


class Server:
    def __init__(self, robot_port=7005, server_port=16000):
        self.robot_port = robot_port
        self.server_port = server_port
        pass

    def start_server(self):
        cmd = "player ./world/simple.cfg -p "+str(self.robot_port)
        self._process_player = subprocess.Popen(cmd, shell=True)
        time.sleep(2)

        pid = subprocess.getoutput("lsof -i :"+str(self.robot_port)+"|  grep player | awk '{print $2}'").split('\n')[0]
        if pid is not '':
            self.player_pid = int(pid)
        print("\033[92m player server pid : {} \033[00m".format(self.player_pid))

        cmd = "python2 ./rpc_server.py "+str(self.server_port)+" "+str(self.robot_port)
        self._process_server = subprocess.Popen(cmd, shell=True)
        time.sleep(2)
        pid = subprocess.getoutput("lsof -i :"+str(self.server_port)+"| grep python2 | awk '{print $2}'")
        if pid is not '':
            self.server_pid = int(pid)
        print("\033[92m rpc server pid : {} \033[00m".format(self.server_pid))
        

    def is_player_alive(self):
        import psutil
        p = psutil.Process(self.player_pid)
        return p.is_running()
        

    def is_rpc_alive(self):
        import psutil
        p = psutil.Process(self.server_pid)
        return p.is_running()
        
    def stop_server(self):
        if self.is_rpc_alive() and subprocess.call(['kill', str(self.server_pid)]) == 0:
            print(" killed {} ".format(self.server_pid))
        if self.is_player_alive() and subprocess.call(['kill', str(self.player_pid)]) == 0:
            print(" killed {} ".format(self.player_pid))

        if self._process_server.returncode is None:
            self._process_server.terminate()
        if self._process_player.returncode is None:
            self._process_player.terminate()

    def __del__(self):
        self.stop_server()
        