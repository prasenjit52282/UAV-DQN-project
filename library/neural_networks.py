import numpy as np
import tensorflow as tf

class my_Q_Network(tf.keras.Model):
    def __init__(self,max_uavs=10,state_size=(80,80,2),action_size=5*5*5,trainable=True):
        super(my_Q_Network,self).__init__()
        self.max_uavs=max_uavs
        self.action_size=action_size
        self.state_size=state_size
        #assuming state=(64,64)

        self.input1=tf.keras.layers.InputLayer(input_shape=state_size) #image
        self.input2=tf.keras.layers.InputLayer(input_shape=((self.max_uavs-1)*8,)) #others
        self.input3=tf.keras.layers.InputLayer(input_shape=(8,)) #self

        self.layer1=tf.keras.layers.Conv2D(32, 5, strides=(1, 1), padding='same',activation="relu")
        self.max_pool1=tf.keras.layers.MaxPool2D(3, strides=3, padding='same') #N,27,27,32

        self.layer2=tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding='same',activation="relu")
        self.max_pool2=tf.keras.layers.MaxPool2D(3, strides=3, padding='same') #N,9,9,64

        self.layer3=tf.keras.layers.Conv2D(256, 3, strides=(1, 1), padding='same',activation="relu")
        self.max_pool3=tf.keras.layers.MaxPool2D(2, strides=2, padding='same') #N,5,5,256

        self.global_avg=tf.keras.layers.GlobalAveragePooling2D() #N,256

        self.concat=tf.keras.layers.Concatenate()

        self.layer4= tf.keras.layers.Dense(500,activation="relu") #N,500
        self.reshape=tf.keras.layers.Reshape((5,5,-1)) #Nx5x5x20
        self.layer5=tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding='same',activation="relu") #Nx5x5x32
        self.layer6=tf.keras.layers.Conv2D(5, 3, strides=(1, 1), padding='same')#Nx5x5x5
        self.q_val= tf.keras.layers.Flatten()
        #assert self.q_val.shape[1]==self.action_size, "action size does not match"
        self.trainable=trainable
#         self.build((None,*self.state_size)) 
        
    def call(self,img,other=None,own=None,a=None):
        inp1=self.input1(img)
        inp2=self.input2(other)
        inp3=self.input3(own)

        net=self.layer1(inp1)
        net=self.max_pool1(net)

        net=self.layer2(net)
        net=self.max_pool2(net)

        net=self.layer3(net)
        net=self.max_pool3(net)

        net=self.global_avg(net)

        net=self.concat([net,inp2,inp3])
        net=self.layer4(net)
        net=self.reshape(net)
        net=self.layer5(net)
        net=self.layer6(net)
        net=self.q_val(net)
        if a is None:
            #sending all q_values
            return net
        else:
            #masking q_values w.r.to actions
            a=tf.keras.utils.to_categorical(a,num_classes=self.action_size)
            net=tf.multiply(a,net)
            return net
   

def Q_Network(max_uavs=10,state_size=(80,80,2),action_size=5*5*5,trainable=True):
    dqn=my_Q_Network(max_uavs,state_size,action_size,trainable)
    a=np.random.random((32,80,80,2))
    b=np.random.random((32,72))
    c=np.random.random((32,8))
    dqn(a,b,c) #initilize
    return dqn

     
def soft_update(source,target,tau):
    """
    Parameters
    ----------
    source : my_Q_Network
        main my_Q_Network for Deep Q Learning algorithms.
    target : my_Q_Network
        fixed Q target my_Q_Network for Deep Q Learning algorithms.
    tau : float
        smoothing factor for Q^=TQ+(1-T)Q^ default=0.001.

    Returns
    -------
    Update fixed Q_targets with Q and tau.

    """
    for tar,src in zip(target.variables,source.variables):
        tar.assign(tau*src+(1-tau)*tar,read_value=False)

def hard_update(source,target):
    """
    Parameters
    ----------
    source : my_Q_Network
        main my_Q_Network for Deep Q Learning algorithms.
    target : my_Q_Network
        fixed Q target my_Q_Network for Deep Q Learning algorithms.

    Returns
    -------
    Update fixed Q target from Q.

    """
    for tar,src in zip(target.variables,source.variables):
        tar.assign(src,read_value=False)