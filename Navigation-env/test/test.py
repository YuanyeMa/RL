import sys
sys.path.append(r"./../")

from EnvNav.Env import RobotWorld
from tqdm import tqdm

env1 = RobotWorld(index=0)
# env2 = RobotWorld(index=1)
# env3 = RobotWorld(index=2)

try:
    for i in tqdm(range(10)):
        done1 = False
        # done2 = done3 =False
        s1 = env1.reset()
        # s2 = env2.reset()
        # s3 = env3.reset()

        steps = 0
        while steps<20:
            steps+=1
            action1 = env1.action_space.sample()
            # action2 = env2.action_space.sample()
            # action3 = env3.action_space.sample()

            s_1, r1, done1, _ = env1.step(action1)
            print(s_1)
            print(r1)
            print(done1)
            # s_2, r2, done2, _ = env2.step(action2)
            # s_3, r3, done3, _ = env3.step(action3)

            s1 = s_1
            # s2 = s_2
            # s3 = s_3
            if done1:
                s1 = env1.reset()
                break
            # if done2:
            #     s2 = env2.reset()
            # if done3:
            #     s3 = env3.reset()
            # if done1 or done2 or done3:
            #     break    
finally:
    env1.close()
    # env2.close()
    # env3.close()
