import tensorflow as tf
import numpy as np
from .memory import replayBuffer
from .neural_networks import Q_Network,hard_update

class DQN:
    def __init__(self,state_size,action_size,memory_size,gamma=0.99):
        
        self.Q=Q_Network(state_size, action_size)
        self.fixed_Q=Q_Network(state_size, action_size,False)
        hard_update(self.Q, self.fixed_Q)
        
        self.optimizer=tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.loss=tf.keras.losses.MeanSquaredError()
        self.gamma=gamma
        
        self.memory=replayBuffer(capacity=memory_size)
        
    def getAction(self,s,episilon):
        #has to be vectorized....to give actions for Nxstate
        if np.random.random()<(1-episilon):
            action=np.argmax(self.Q(s.reshape(1,-1)),axis=1)[0]
        else:
            action=np.random.randint(self.Q.action_size)
        return action
        
        
    def learn(self,batch_size):
        s,a,r,s_,nd=self.memory.sample(batch_size)
        
        TD_target=r+self.gamma*np.multiply(nd,np.max(self.fixed_Q(s_),axis=1))
        a_map=tf.keras.utils.to_categorical(a,self.Q.action_size)
        Q_target=np.multiply(TD_target.reshape(-1,1),a_map)
        
        with tf.GradientTape() as tape:
            loss=self.loss(Q_target,self.Q(s,a))
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
