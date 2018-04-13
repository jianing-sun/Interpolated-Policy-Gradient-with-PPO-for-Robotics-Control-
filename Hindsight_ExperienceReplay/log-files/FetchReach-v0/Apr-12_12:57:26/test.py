import gym

env = gym.make("FetchReach-v0")
obs = env.reset()
print(env.action_space.shape[0])
print(obs['achieved_goal'])
print(obs)
# print(env.observation_space.shape[0])
