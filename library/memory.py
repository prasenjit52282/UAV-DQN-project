import random
import numpy as np
from collections import namedtuple,deque

class replayBuffer:
    transition=namedtuple('Transition', ['current_state', 'action','reward','next_state','not_done'])
    def __init__(self,capacity,env):
        self.capacity=capacity
        self.env=env
        self.queue=deque([],maxlen=self.capacity)
        
    def push(self,s,a,r,s_,nd):
        #may store images as int8 -->replace float32 to int8
        self.queue.append(replayBuffer.transition(s,a,r,s_,nd))
        
    def sample(self,batch_size):
        transition_batch=random.choices(self.queue,k=batch_size)
        
        current_state_img_batch=[]
        current_state_other_batch=[]
        current_state_own_batch=[]
        action_batch=[]
        reward_batch=[]
        next_state_img_batch=[]
        next_state_other_batch=[]
        next_state_own_batch=[]
        not_done_batch=[]
        
        for tr in transition_batch:
            img,other,own=self.env.state_formatter.get_state(tr.current_state)
            current_state_img_batch.extend(img.tolist())
            current_state_other_batch.extend(other.tolist())
            current_state_own_batch.extend(own.tolist())

            img,other,own=self.env.state_formatter.get_state(tr.next_state)
            next_state_img_batch.extend(img.tolist())
            next_state_other_batch.extend(other.tolist())
            next_state_own_batch.extend(own.tolist())

            action_batch.extend(tr.action)
            reward_batch.extend(tr.reward)
            dones=[tr.not_done]*len(tr.action)
            not_done_batch.extend(dones)

        #may need to rescale images as float32 just /255.
        return (np.array(current_state_img_batch),
                np.array(current_state_other_batch),
                np.array(current_state_own_batch),
                np.array(action_batch),
                np.array(reward_batch),
                np.array(next_state_img_batch),
                np.array(next_state_other_batch),
                np.array(next_state_own_batch),
                np.uint8(not_done_batch))