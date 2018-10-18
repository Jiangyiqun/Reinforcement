import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
wait = input("PRESS ANY KEY TO CONTINUE.")