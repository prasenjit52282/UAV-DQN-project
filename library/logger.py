import datetime
import tensorflow as tf

import sys
sys.path.append("../")

class TensorboardLogger:
    def __init__(self,loc="./logs",experiment="DQN"):
        self.base_log_dir=loc
        self.experiment_name=experiment
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.base_log_dir + current_time + self.experiment_name
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def log(self,step,metrics):
        with self.summary_writer.as_default():
            for k,v in metrics.items():
                tf.summary.scalar(f'{self.experiment_name}/{k}',v,step=step)