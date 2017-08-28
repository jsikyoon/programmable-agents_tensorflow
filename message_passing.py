import tensorflow as tf 
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 40;
LAYER2_SIZE = 40;
eps=1e-3;

class Message_passing:
    def __init__(self,sess,state_dim,obj_num,fea_size,p,state_input,postfix):
                self.sess = sess;
                self.state_dim = state_dim;
                self.fea_size=fea_size;
                self.obj_num=obj_num;
                self.p=p;
                self.state_input=state_input;
                self.postfix=postfix;
                # create Message Passing nets
                self.state_output,self.net = self.create_network();
                self.params_num=len(self.net);

    def create_network(self):
                layer1_size = LAYER1_SIZE;
                layer2_size = LAYER2_SIZE;
                state_input2 = tf.transpose(tf.reshape(self.state_input,[-1,self.obj_num,self.fea_size]),[0,2,1]);
                state_input2 = tf.unstack(state_input2,self.obj_num,2);
                # local transform function
                f_out=np.zeros(self.obj_num,dtype=object);
                with tf.variable_scope('message_passing_f_'+self.postfix):
                  w1=tf.get_variable('w1',shape=[self.fea_size,layer1_size]);
                  b1=tf.get_variable('b1',shape=[layer1_size]);
                  w2=tf.get_variable('w2',shape=[layer1_size,layer2_size]);
                  b2=tf.get_variable('b2',shape=[layer2_size]);
                  w3=tf.get_variable('w3',shape=[layer2_size,self.fea_size]);
                  b3=tf.get_variable('b3',shape=[self.fea_size]);
                for i in range(len(state_input2)): 
                  with tf.variable_scope('message_passing_f_'+self.postfix,reuse=True):
                    layer1=tf.nn.relu(tf.matmul(state_input2[i],w1)+b1);
                    layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2);
                    f_out[i]=tf.tanh(tf.matmul(layer2,w3)+b3);
                self.f_out_tmp=tf.stack(list(f_out),1);
                # interaction network
                r_out=np.zeros((self.obj_num,self.obj_num),dtype=object);
                with tf.variable_scope('message_passing_r_'+self.postfix):
                  w1=tf.get_variable('w1',shape=[self.fea_size*2,layer1_size]);
                  b1=tf.get_variable('b1',shape=[layer1_size]);
                  w2=tf.get_variable('w2',shape=[layer1_size,layer2_size]);
                  b2=tf.get_variable('b2',shape=[layer2_size]);
                  w3=tf.get_variable('w3',shape=[layer2_size,self.fea_size]);
                  b3=tf.get_variable('b3',shape=[self.fea_size]);
                r_in=np.zeros((self.obj_num,self.obj_num),dtype=object);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                      r_in[i,j]=tf.concat([state_input2[i],state_input2[j]],1);
                      with tf.variable_scope('message_passing_r_'+self.postfix,reuse=True):
                        layer1=tf.nn.relu(tf.matmul(r_in[i,j],w1)+b1);
                        layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2);
                        r_out[i,j]=tf.tanh(tf.matmul(layer2,w3)+b3);
                # get alpha
                alpha=np.zeros((self.obj_num,self.obj_num),dtype=object);
                with tf.variable_scope('message_passing_c_'+self.postfix):
                  w1=tf.get_variable('w1',shape=[self.fea_size,self.fea_size]);
                  b1=tf.get_variable('b1',shape=[self.fea_size]);
                with tf.variable_scope('message_passing_q_'+self.postfix):
                  w1=tf.get_variable('w1',shape=[self.fea_size,self.fea_size]);
                  b1=tf.get_variable('b1',shape=[self.fea_size]);
                c=np.zeros(self.obj_num,dtype=object);
                q=np.zeros(self.obj_num,dtype=object);
                for i in range(self.obj_num):
                  with tf.variable_scope('message_passing_c_'+self.postfix):
                    c[i]=tf.matmul(state_input2[i],w1)+b1;
                  with tf.variable_scope('message_passing_q_'+self.postfix):
                    q[i]=tf.matmul(state_input2[i],w1)+b1;
                alpha_hat=np.zeros((self.obj_num,self.obj_num),dtype=object);
                p_list=tf.unstack(self.p,self.obj_num,1);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                      alpha_hat[i,j]=tf.multiply(state_input2[i],tf.tanh(q[i]+c[j]));
                      alpha_hat[i,j]=tf.reduce_sum(alpha_hat[i,j],1);
                      alpha_hat[i,j]=p_list[j]*tf.exp(alpha_hat[i,j]);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                        alpha[i,j]=alpha_hat[i,j]/(np.sum(alpha_hat[i,:]));
                # new state from message_passing
                state_output=np.zeros(self.obj_num,dtype=object);
                for i in range(self.obj_num):
                  state_output[i]=f_out[i];
                  for j in range(self.obj_num):
                    if(i!=j):
                      state_output[i]+=tf.multiply(tf.stack([alpha[i,j]]*self.fea_size,1),r_out[i,j]);
                state_output=tf.stack(list(state_output),1);
                state_output=tf.reshape(state_output,[-1,self.state_dim]);
                
                f_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='message_passing_f_'+self.postfix);
                r_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='message_passing_r_'+self.postfix);
                c_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='message_passing_c_'+self.postfix);
                q_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='message_passing_q_'+self.postfix);
                params_list=f_params+r_params+c_params+q_params;
               
                return state_output,params_list;
    
    def run_target_nets(self,state_input,p,params_list):
                idx=0;
                state_input2 = tf.transpose(tf.reshape(state_input,[-1,self.obj_num,self.fea_size]),[0,2,1]);
                state_input2 = tf.unstack(state_input2,self.obj_num,2);
                # f funtion
                f_out=np.zeros(self.obj_num,dtype=object);
                for i in range(len(state_input2)): 
                  layer1=tf.nn.relu(tf.matmul(state_input2[i],params_list[idx])+params_list[idx+1]);
                  layer2=tf.nn.relu(tf.matmul(layer1,params_list[idx+2])+params_list[idx+3]);
                  f_out[i]=tf.tanh(tf.matmul(layer2,params_list[idx+4])+params_list[idx+5]);
                idx+=6;
                # r function
                r_in=np.zeros((self.obj_num,self.obj_num),dtype=object);
                r_out=np.zeros((self.obj_num,self.obj_num),dtype=object);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                      r_in[i,j]=tf.concat([state_input2[i],state_input2[j]],1);
                      layer1=tf.nn.relu(tf.matmul(r_in[i,j],params_list[idx])+params_list[idx+1]);
                      layer2=tf.nn.relu(tf.matmul(layer1,params_list[idx+2])+params_list[idx+3]);
                      r_out[i,j]=tf.tanh(tf.matmul(layer2,params_list[idx+4])+params_list[idx+5]);
                idx+=6;
                # get alpha
                alpha=np.zeros((self.obj_num,self.obj_num),dtype=object);
                c=np.zeros(self.obj_num,dtype=object);
                q=np.zeros(self.obj_num,dtype=object);
                for i in range(self.obj_num):
                  c[i]=tf.matmul(state_input2[i],params_list[idx])+params_list[idx+1];
                  q[i]=tf.matmul(state_input2[i],params_list[idx+2])+params_list[idx+3];
                idx+=4;
                alpha_hat=np.zeros((self.obj_num,self.obj_num),dtype=object);
                p_list=tf.unstack(p,self.obj_num,1);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                      alpha_hat[i,j]=tf.multiply(state_input2[i],tf.tanh(q[i]+c[j]));
                      alpha_hat[i,j]=tf.reduce_sum(alpha_hat[i,j],1);
                      alpha_hat[i,j]=p_list[j]*tf.exp(alpha_hat[i,j]);
                for i in range(self.obj_num):
                  for j in range(self.obj_num):
                    if(i!=j):
                      alpha[i,j]=alpha_hat[i,j]/np.sum(alpha_hat[i,:]);
                # new state from message_passing
                state_output=np.zeros(self.obj_num,dtype=object);
                for i in range(self.obj_num):
                  state_output[i]=f_out[i];
                  for j in range(self.obj_num):
                    if(i!=j):
                      state_output[i]+=tf.multiply(tf.stack([alpha[i,j]]*self.fea_size,1),r_out[i,j]);
                state_output=tf.stack(list(state_output),1);
                state_output=tf.reshape(state_output,[-1,self.state_dim]);
                return state_output;

        
