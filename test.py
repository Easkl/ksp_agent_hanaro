from kspdg.pe1.e1_envs import PE1_E1_I3_Env
import time

env = PE1_E1_I3_Env()

obs = env.reset()
done = False
while not done:
    print(obs)
    time.sleep(1)