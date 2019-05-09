基于playerc写的一个避障导航的仿真环境  

playerc version: v3.1.1  
此外还用到了python2的xmlrpclib做RPC(remote process call) server  

调用关系： env->robot->rpc_client -> rpc_server->playerc  
