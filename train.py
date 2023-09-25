from ces import *
from cmaes import *
from cmamrl import CMAMRLAgent
from cmrl import MRLAgent

# agent = CESAgent("CartPole-v0")

# agent = CMAESAgent("MountainCar-v0")


# agent = CMAMRLAgent("LunarLander-v2")

agent = MRLAgent("LunarLander-v2")

rewards = agent.train(num_run=400)

