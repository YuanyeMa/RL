from baselines.ddpg import ddpg
#  baselines.common.vec_env.dummy_vec_env.DummyVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.cmd_util import make_env
import gym
import sys
sys.path.append(r"./../")

from EnvNav.Env import RobotWorld

if __name__ == "__main__":
    env = RobotWorld(index=0)
    # env =  gym.make("Pendulum-v0")
    # env_id  = "Pendulum-v0"
    # env = make_env(env_id = env_id, env_type=None)
    env = DummyVecEnv([lambda: env])
    print(env.action_space)
    act = ddpg.learn(env=env, network = "mlp" , total_timesteps=10000)

    print("Finish!")
