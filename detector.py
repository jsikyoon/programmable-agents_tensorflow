import tensorflow as tf 
import numpy as np
import math

class Detector:
    def __init__(self,sess,state_dim,obj_num,fea_size,state_input,postfix):
                self.sess = sess;
                self.state_dim = state_dim;
                self.fea_size=fea_size;
                self.obj_num=obj_num;
                self.postfix=postfix;
                self.state_input=state_input;

                # create detector
                self.Theta,self.net = self.create_network();
                self.params_num=len(self.net);

    def create_network(self):
                state_input2 = tf.reshape(self.state_input,[-1,self.fea_size]);
                # 6 Detectors
                output=np.zeros(6,dtype=object);
                params_list=[];
                for i in range(6):
                  with tf.variable_scope('detector_'+str(i+1)+"_"+self.postfix):
                    w=tf.get_variable('w',shape=[self.fea_size,1]);
                    b=tf.get_variable('b',shape=[1]);
                    output[i]=tf.sigmoid(tf.matmul(state_input2,w)+b);
                  params_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector_'+str(i+1)+"_"+self.postfix);
                output=tf.concat(list(output),1);
                
                """
                state_input3 = tf.reshape(self.state_input,[-1,self.obj_num,self.fea_size]);  
                state_input3 = tf.unstack(state_input3,self.fea_size,2);
                output = tf.stack(state_input3[5:14],2);
                """

                output=tf.reshape(output,[-1,self.obj_num*6]);
                return output,params_list;

    def run_target_nets(self,state_input,params_list):    
                state_input2 = tf.reshape(state_input,[-1,self.fea_size]);
                output=np.zeros(6,dtype=object);
                idx=0;
                for i in range(6):
                  output[i]=tf.sigmoid(tf.matmul(state_input2,params_list[idx])+params_list[idx+1]);
                  idx+=2;
                output=tf.concat(list(output),1);

                """
                state_input3 = tf.reshape(state_input,[-1,self.obj_num,self.fea_size]);  
                state_input3 = tf.unstack(state_input3,self.fea_size,2);
                output = tf.stack(state_input3[5:14],2);
                """

                output=tf.reshape(output,[-1,self.obj_num*6]);
                return output;
