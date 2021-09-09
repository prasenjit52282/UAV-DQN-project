import tensorflow as tf

class Q_Network(tf.keras.Model):
    def __init__(self,state_size,action_size,trainable=True):
        super(Q_Network,self).__init__()
        self.action_size=action_size
        self.state_size=state_size
        #assuming state=(64,64)
        self.layer1=tf.keras.layers.Conv2D(32, 5, strides=(1, 1), padding='same',activation="relu")
        self.max_pool1=tf.keras.layers.MaxPool2D(3, strides=3, padding='same') #N,22,22,32

        self.layer2=tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same',activation="relu")
        self.max_pool2=tf.keras.layers.MaxPool2D(2, strides=2, padding='same') #N,11,11,64

        self.layer3=tf.keras.layers.Conv2D(256, 3, strides=(1, 1), padding='same',activation="relu")
        self.max_pool3=tf.keras.layers.MaxPool2D(2, strides=2, padding='same') #N,6,6,256

        self.global_avg=tf.keras.layers.GlobalAveragePooling2D() #N,256

        self.layer4= tf.keras.layers.Dense(256,activation="relu") #N,256
        self.layer5= tf.keras.layers.Dense(action_size) #N,no_of_act=15x15=225
                                                        #(if we consider 15x15 grid to move)
        
        self.trainable=trainable
        self.build((None,self.state_size)) 
        
    def call(self,s,a=None):
        net=self.layer1(s)
        net=self.max_pool1(net)

        net=self.layer2(net)
        net=self.max_pool2(net)

        net=self.layer3(net)
        net=self.max_pool3(net)

        net=self.global_avg(net)

        net=self.layer4(net)
        net=self.layer5(net)
        if a is None:
            #sending all q_values
            return net
        else:
            #masking q_values w.r.to actions
            a=tf.keras.utils.to_categorical(a,num_classes=self.action_size)
            net=tf.multiply(a,net)
            return net
        
def soft_update(source,target,tau):
    """
    Parameters
    ----------
    source : Q_Network
        main Q_network for Deep Q Learning algorithms.
    target : Q_network
        fixed Q target Q_network for Deep Q Learning algorithms.
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
    source : Q_Network
        main Q_network for Deep Q Learning algorithms.
    target : Q_network
        fixed Q target Q_network for Deep Q Learning algorithms.

    Returns
    -------
    Update fixed Q target from Q.

    """
    for tar,src in zip(target.variables,source.variables):
        tar.assign(src,read_value=False)