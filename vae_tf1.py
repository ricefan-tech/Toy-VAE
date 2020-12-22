import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from tqdm.auto import tqdm
import tensorflow as tf

class VAE(object):
    """Variational Auto Encoder (VAE)."""

    def __init__(self, n_latent, n_hidden=50, alpha=0.5):
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha = alpha
    
    #activation function
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    def encoder(self, X_in, input_dim):
        with tf.variable_scope("encoder", reuse=None):
            x = X_in
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*2, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*3, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*4, activation=self.lrelu)
            mn = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            sd = tf.layers.dense(x, units=self.n_latent, activation=self.lrelu)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
            z = mn + tf.multiply(epsilon, tf.exp(sd / 2.))
            return z, mn, sd
    
    def decoder(self, sampled_z, input_dim):
        with tf.variable_scope("decoder", reuse=None):
            x = sampled_z
            x = tf.layers.dense(x, units=self.n_hidden, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*2, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*3, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.n_hidden*4, activation=self.lrelu)
            x = tf.layers.dense(x, units=input_dim, activation=tf.nn.sigmoid)
            x = tf.reshape(x, shape=[-1, input_dim])
            return x

    def train(self, data, n_epochs=10000, learning_rate=0.005,
              show_progress=False):
        data = utils.as_float_array(data)
        
        #scale input data
        assert data.max() <= 1. and data.min() >=0., \
            "All features of the dataset must be between 0 and 1."
        tf.reset_default_graph()
        input_dim = data.shape[1]
        X_in = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="Y")
        Y_flat = Y
        
        self.sampled, mn, sd = self.encoder(X_in,input_dim=input_dim)
        self.dec = self.decoder(self.sampled, input_dim=input_dim)
        
        #reshape decoder output
        unreshaped = tf.reshape(self.dec, [-1, input_dim])
        decoded_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), 1)
        self.loss = tf.reduce_mean( (1-self.alpha)*10000*decoded_loss + self.alpha * latent_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for i in range(1,n_epochs+1):
            self.sess.run(optimizer, feed_dict={X_in: data,  Y: data})
            if not i % 100 and show_progress:
                ls, recon, KL, d = self.sess.run([self.loss,tf.reduce_mean(decoded_loss), tf.reduce_mean(latent_loss), self.dec], feed_dict={X_in: data, Y: data})
                
                #choose random axes to output scatterplot of real data vs decoder output
                projections = np.random.randint(0, data.shape[1], size=2)
                plt.scatter(data[:, projections[0]], data[:, projections[1]])
                plt.scatter(d[:, projections[0]], d[:, projections[1]])
                plt.xlabel(str(projections[0]))
                plt.ylabel(str(projections[1]))
                plt.show()
        self.samp_eval=self.sampled.eval(session=self.sess,feed_dict={X_in:data})   
    
    #generate new samples from random noise
    def generate(self,  n_samples=None):
        if n_samples is not None:
            randoms = np.random.normal(0, 1, size=(n_samples, self.n_latent))
        else:
            randoms = np.random.normal(0, 1, size=(1, self.n_latent))
        samples = self.sess.run(self.dec, feed_dict={self.sampled: randoms})
        if n_samples is None:
            return samples[0]
        return samples

