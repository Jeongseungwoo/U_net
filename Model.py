import tensorflow as tf
import datetime, os
from U_net import U_net

log_dir = "./log"

class Model(object) :
    def __init__(self, mode = "train", network = U_net, log_dir = log_dir) :
        self.mode = mode 
        self.network = network

        self.build_model()
        self.optimizer_initializer()
        
        self.saver = tf.train.Saver()  # save checkpoint
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        if self.mode == 'train':
            self.train_step = 0
            now = datetime.datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))      
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.U_net_summary = self.summary()
        
    def build_model(self) :

        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, shape = [None, 572, 572, 2]) # Batch_size, w, h, c 
        self.labels = tf.placeholder(tf.float32, shape = [None, 388, 388, 2])
        
        self.outputs = self.network(self.inputs, name = "U_net")
        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits=self.outputs))

    def optimizer_initializer(self) :
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    
    def train(self, inputs, labels) :
        outputs, U_net_loss, U_net_summary, _ = self.sess.run([self.outputs, self.loss, self.U_net_summary,self.optimizer],
                                                               feed_dict = {self.inputs : inputs, self.labels : labels})
        self.writer.add_summary(self.U_net_summary, self.train_step)
        
        self.train_step  += 1
        return U_net_loss
    
    def test(self, inputs) :
        outputs = self.sess.run(self.outputs, feed_dict = {self.inputs : inputs})
        return outputs 
    
    def save(self, directory, filename) :
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)

    def summary(self) :
        with tf.name_scope("Unet_summary") :
            U_net_summary = tf.summary.scalar("Unet_loss",self.loss)
        return U_net_summary
    
if __name__ == "__main__" :
    model = Model()