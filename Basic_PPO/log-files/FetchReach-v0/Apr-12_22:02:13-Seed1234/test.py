import gym

env = gym.make('FetchReach-v0')
ob = env.reset()
print(ob)
env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal', 'achieved_goal'])
print(env.observation_space.shape[0])
print(env.action_space.shape[0])

while True:
    env.render()
