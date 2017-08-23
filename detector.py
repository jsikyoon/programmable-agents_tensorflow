import tensorflow as tf 
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 40;
LAYER2_SIZE = 40;

class Detector:
	def __init__(self,sess,state_dim,fea_size,obj_num):
                self.sess = sess;
                self.state_dim = state_dim;
                self.fea_size=fea_size;
                self.obj_num=obj_num;

                # create detector
                self.state_input,self.Theta,self.net = self.create_network();

	def create_network(self):
                layer1_size = LAYER1_SIZE;
                layer2_size = LAYER2_SIZE;
                state_input = tf.placeholder("float",[None,self.state_dim]);
                state_input2 = tf.reshape(state_input,[-1,self.fea_size]);
                # 9 Detectors
                output=np.zeros(9,dtype=object);
                params_list=[];
                for i in range(9):
                  with tf.variable_scope('detector_'+str(i+1)):
                    w1=tf.get_variable('w1',shape=[self.fea_size,layer1_size]);
                    b1=tf.get_variable('b1',shape=[layer1_size]);
                    w2=tf.get_variable('w2',shape=[layer1_size,layer2_size]);
                    b2=tf.get_variable('b2',shape=[layer2_size]);
                    w3=tf.get_variable('w3',shape=[layer2_size,self.fea_size]);
                    b3=tf.get_variable('b3',shape=[self.fea_size]);
                    layer1=tf.nn.relu(tf.matmul(state_input2,w1)+b1);
                    layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2);
                    output[i]=tf.tanh(tf.matmul(layer2,w3)+b3);
                  params_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector_'+str(i+1));
                output=tf.concat(list(output),1);
                output=tf.reshape(output,[-1,self.fea_size*9]);
                
                return state_input,output,params_list;
		
