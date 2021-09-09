#%%Imports
import gym
from library.algorithms import DQN,get_episilon
from library.logger import TensorboardLogger

#%%Initilize Algorithm and Environment
env=gym.Env(Our_Environment) #Our Simulator
env._max_episode_steps=#max_time_in_the_env
agent=DQN(env.observation_space.shape[0],env.action_space.n,memory_size=100000,gamma=0.99)
logger=TensorboardLogger(loc="./logs",experiment="DQN")

#%%filling replayBuffer with 50000 transitions
curr_state=env.reset()
for _ in range(50000):
    #loop through all agents and sample actions=(#agents,).....curr_state=(#agent, h, w, stack)~=next_state
    act=env.action_space.sample()
    next_state,reward,done,info=env.step(act)
    agent.memory.push(curr_state, act, reward, next_state, not done)
    if done:
        curr_state=env.reset()
    else:
        curr_state=next_state

#%%Training for 10 lack steps with linear deacy exploration
episode=0
episode_reward=0

curr_state=env.reset()
for step in range(1,1000000+1):
    #loop through all agents.....curr_state=(#agent, h, w, stack)~=next_state
    #we will assume that uavs took the same time to reach to the next target so velocity will be different.
    episilon=get_episilon(step,0.01,100000)
    act=agent.getAction(curr_state,episilon)
    next_state,reward,done,info=env.step(act)
    episode_reward+=reward
    agent.memory.push(curr_state, act, reward, next_state, not done)
    if done:
        episode+=1
        print('On Episode {} reward {} on global_step {}'.format(episode,episode_reward,step))
        logger.log(step,{'episode_reward':episode_reward,"episilon":episilon}) #tensorboard logging
        episode_reward=0
        curr_state=env.reset()
    else:
        curr_state=next_state
     
    if step%10000==0:
        agent.updateFixedQ()
           
    if step%4==0:
        agent.learn(batch_size=32)
    

#%%Testing for 10 episodes
for _ in range(10):
    curr_state=env.reset()
    done=False
    episode_reward=0
    while not done:
        env.render()
        act=agent.getAction(curr_state, 0.01)
        next_state,reward,done,info=env.step(act)
        episode_reward+=reward
        curr_state=next_state
    print('On episode {} reward {}'.format(_,episode_reward))
env.close()
