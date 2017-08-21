import tensorflow as tf 
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 40;
LAYER2_SIZE = 40;
LEARNING_RATE = 1e-4;
fea_size=18;

class Detector:
	def __init__(self,sess,state_dim):
                self.sess = sess;
                self.state_dim = state_dim;
                self.fea_size=fea_size;
                self.obj_num=int(self.state_dim/fea_size);

                # create detector
                self.state_input,self.Theta,self.net = self.create_network(state_dim);
                # define training rules
                self.create_training_method()
                self.sess.run(tf.global_variables_initializer())

	def create_training_method(self):
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE);

	def create_network(self,state_dim):
                layer1_size = LAYER1_SIZE;
                layer2_size = LAYER2_SIZE;
                state_input = tf.placeholder("float",[None,state_dim]);
                state_input2 = tf.reshape(state_input,[-1,fea_size]);
                # 9 Detectors, 2 hidden layers and 1 output layer
                w1=np.zeros(9,dtype=object);
                b1=np.zeros(9,dtype=object);
                w2=np.zeros(9,dtype=object);
                b2=np.zeros(9,dtype=object);
                w3=np.zeros(9,dtype=object);
                b3=np.zeros(9,dtype=object);
                layer1=np.zeros(9,dtype=object);
                layer2=np.zeros(9,dtype=object);
                output=np.zeros(9,dtype=object);
                for i in range(9):
                  w1[i] = self.variable([fea_size,layer1_size],fea_size);
                  b1[i] = self.variable([layer1_size],fea_size);
                  w2[i] = self.variable([layer1_size,layer2_size],layer1_size);
                  b2[i] = self.variable([layer2_size],layer1_size);
                  w3[i] = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3));
                  b3[i] = tf.Variable(tf.random_uniform([1],-3e-3,3e-3));
                  layer1[i] = tf.nn.relu(tf.matmul(state_input2,w1[i]) + b1[i]);
                  layer2[i] = tf.nn.relu(tf.matmul(layer1[i],w2[i]) + b2[i]);
                  output[i] = tf.tanh(tf.matmul(layer2[i],w3[i]) + b3[i]);
                output=tf.concat(list(output),1);
                output=tf.reshape(output,[-1,fea_size*9]);
               
                params_list=list(w1)+list(b1)+list(w2)+list(b2)+list(w3)+list(b3);
                return state_input,output,params_list;

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

		
