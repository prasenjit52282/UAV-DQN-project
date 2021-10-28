#%% Imports
import sys
sys.path.append("../")
import numpy as np
from library.algorithms import DQN,get_episilon
from library.logger import TensorboardLogger
from UAVsim.library.wrapper import UAV_sim

#%% Init
print("Initializing .....")
env=UAV_sim(length=300, bredth=300, max_height=150, max_num_uavs=10, base_position=None,
                target_positions=20, mine_positions=10, distruction_range=5, 
                detection_range=15, mine_radar_range=6, max_time=50) #Our Simulator

agent=DQN(env,memory_size=100000,gamma=0.99)
logger=TensorboardLogger(loc="./logs/",experiment="DQN")

#%% Collecting
print("Collecting random transitions .....")
curr_state=env.reset()
for _ in range(100):
    num_acts=len(curr_state["uavs"])
    acts=env.convert_action_arr([np.random.randint(0,125) for a in range(num_acts)])
    next_state,reward,done,info=env.step(acts)
    agent.memory.push(curr_state, acts, reward, next_state, not done)
    if done:
        curr_state=env.reset()
    else:
        curr_state=next_state

#%% Training
episode=0
episode_reward=0
past_episode_start_step=0
print("Training starts .......")
curr_state=env.reset()
for step in range(1,1000000+1):
    episilon=get_episilon(step,0.01,100000)
    act=agent.getAction(curr_state,episilon)
    next_state,reward,done,info=env.step(env.convert_action_arr(act))
    episode_reward+=reward.sum()
    agent.memory.push(curr_state, act, reward, next_state, not done)
    if done:
        episode+=1
        total_step_run=step-past_episode_start_step
        print('On Episode {} reward {:.2f} on global_step {} run for {}'.format(episode,episode_reward,step,total_step_run))
        logger.log(step,{'episode_reward':episode_reward,"episilon":episilon,"run_for_step":total_step_run}) #tensorboard logging
        past_episode_start_step=step
        episode_reward=0
        curr_state=env.reset()
    else:
        curr_state=next_state
     
    if step%10000==0:
        agent.updateFixedQ()
           
    if step%4==0:pass
        #agent.learn(batch_size=32)

#%% Testing   
print("Testing ......")
for _ in range(10):
    curr_state=env.reset()
    done=False
    episode_reward=0
    while not done:
        #env.render()
        act=agent.getAction(curr_state, 0.01)
        next_state,reward,done,info=env.step(env.convert_action_arr(act))
        episode_reward+=reward.sum()
        curr_state=next_state
    print('On episode {} reward {:.2f}'.format(_,episode_reward))
env.close()

# %%
#END