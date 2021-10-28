import tensorflow as tf
import numpy as np
from .memory import replayBuffer
from .neural_networks import Q_Network,hard_update

class DQN:
    def __init__(self,env,memory_size,gamma=0.99):
        
        self.Q=Q_Network()
        self.fixed_Q=Q_Network(trainable=False)
        hard_update(self.Q, self.fixed_Q)
        
        self.optimizer=tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.loss=tf.keras.losses.MeanSquaredError()
        self.gamma=gamma
        self.env=env
        self.memory=replayBuffer(capacity=memory_size,env=self.env)
        
    def getAction(self,s,episilon):
        if np.random.random()<(1-episilon):
            img,other,own=self.env.state_formatter.get_state(s)
            actions=np.argmax(self.Q(img,other,own),axis=1)
        else:
            num_acts=len(s["uavs"])
            actions=[np.random.randint(0,self.Q.action_size) for a in range(num_acts)]
        return actions
        
        
    def learn(self,batch_size): #not migrated yet
        s_img,s_other,s_own,a,r,s__img,s__other,s__own,nd=self.memory.sample(batch_size)
        
        TD_target=r+self.gamma*np.multiply(nd,np.max(self.fixed_Q(s__img,s__other,s__own),axis=1))
        a_map=tf.keras.utils.to_categorical(a,self.Q.action_size)
        Q_target=np.multiply(TD_target.reshape(-1,1),a_map)
        
        with tf.GradientTape() as tape:
            loss=self.loss(Q_target,self.Q(s_img,s_other,s_own,a))
        grads=tape.gradient(loss,self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.Q.trainable_variables))
        
    def updateFixedQ(self):
        hard_update(self.Q, self.fixed_Q)
        


def get_episilon(step,steady_episilon=0.01,steady_step=10000):
    if step>steady_step:
        return steady_episilon
    else:
        m=(steady_episilon-1)/steady_step
        return m*step+1
