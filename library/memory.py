import random
import numpy as np
from collections import namedtuple,deque

class replayBuffer:
    transition=namedtuple('Transition', ['current_state', 'action','reward','next_state','not_done'])
    def __init__(self,capacity):
        self.capacity=capacity
        self.queue=deque([],maxlen=self.capacity)
        
    def push(self,s,a,r,s_,nd):
        #may store images as int8 -->replace float32 to int8
        self.queue.append(replayBuffer.transition(np.float32(s),a,r,np.float32(s_),nd))
        
    def sample(self,batch_size):
        transition_batch=random.choices(self.queue,k=batch_size)
        
        current_state_batch=[]
        action_batch=[]
        reward_batch=[]
        next_state_batch=[]
        not_done_batch=[]
        
        for tr in transition_batch:
            current_state_batch.append(tr.current_state)
            action_batch.append(tr.action)
            reward_batch.append(tr.reward)
            next_state_batch.append(tr.next_state)
            not_done_batch.append(tr.not_done)

        #may need to rescale images as float32 just /255.
        return (np.array(current_state_batch),
                np.array(action_batch),
                np.array(reward_batch),
                np.array(next_state_batch),
                np.uint8(not_done_batch))