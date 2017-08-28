
import tensorflow as tf 
import numpy as np
import math
from detector import Detector
from message_passing import Message_passing
from program import Program

# Parameters
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01
order_num=2

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,obj_num,fea_size,action_dim):
        self.time_step = 0;
        self.sess = sess;
        self.state_dim=state_dim;
        self.obj_num=obj_num;
        self.fea_size=fea_size;
        self.action_dim=action_dim;
        self.order_num=order_num;
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net,\
        self.program_order= self.create_q_network(state_dim,action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update,\
        self.target_program_order = self.create_target_q_network(state_dim,action_dim,self.net)

        self.create_training_method()

        # initialization 
        self.sess.run(tf.global_variables_initializer())
            
        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self,state_dim,action_dim):
        state_input = tf.placeholder("float",[None,state_dim])
        program_order = tf.placeholder("float",[None,3]);
        # Detector
        self.detector=Detector(self.sess,self.state_dim,self.obj_num,self.fea_size,state_input,"critic");
        d_params=self.detector.net;
        # Program
        self.program=Program(self.sess,self.state_dim,self.obj_num,self.fea_size,self.detector.Theta,program_order,"critic");
        p=self.program.p;
        # Message Passing
        self.message_passing=Message_passing(self.sess,self.state_dim,self.obj_num,self.fea_size,self.program.p,state_input,"critic");
        m_params=self.message_passing.net;
        # get h
        Omega_dot=self.message_passing.state_output;
        Omega_dot=tf.reshape(Omega_dot,[-1,self.obj_num,self.fea_size]);
        Omega_dot=tf.transpose(Omega_dot,perm=[0,2,1]);
        Omega_dot=tf.unstack(Omega_dot,self.obj_num,2);
        p_list=tf.unstack(p,self.obj_num,1);
        h=0;
        for i in range(self.obj_num):
          h+=tf.stack([p_list[i]]*self.fea_size,1)*Omega_dot[i];
        # get Q
        action_input = tf.placeholder("float",[None,action_dim]);
        with tf.variable_scope('critic_nets'):
          w1=tf.get_variable('w1',shape=[self.action_dim,self.fea_size]);
          b1=tf.get_variable('b1',shape=[self.fea_size]);
          w2=tf.get_variable('w2',shape=[self.fea_size,1]);
          b2=tf.get_variable('b2',shape=[1]);
          q_value_output=tf.matmul(tf.tanh(h+tf.matmul(action_input,w1)+b1),w2)+b2;
        c_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_nets');
        # param list
        param_list=d_params+m_params+c_params;
        return state_input,action_input,q_value_output,param_list,program_order;

    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        state_input = tf.placeholder("float",[None,state_dim])
        program_order = tf.placeholder("float",[None,3]);
        action_input = tf.placeholder("float",[None,action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        # params for each net
        d_net=net[:self.detector.params_num];
        m_net=net[self.detector.params_num:(self.detector.params_num+self.message_passing.params_num)];
        c_net=net[(self.detector.params_num+self.message_passing.params_num):];
        # run detector
        Theta=self.detector.run_target_nets(state_input,d_net);
        # run program
        p=self.program.run_target_nets(Theta,program_order);
        # run message_passing
        Omega_dot=self.message_passing.run_target_nets(state_input,p,m_net);
        # get h
        Omega_dot=tf.reshape(Omega_dot,[-1,self.obj_num,self.fea_size]);
        Omega_dot=tf.transpose(Omega_dot,perm=[0,2,1]);
        Omega_dot=tf.unstack(Omega_dot,self.obj_num,2);
        p_list=tf.unstack(p,self.obj_num,1);
        h=0;
        for i in range(self.obj_num):
          h+=tf.stack([p_list[i]]*self.fea_size,1)*Omega_dot[i];
        # get Q  
        q_value_output=tf.matmul(tf.tanh(h+tf.matmul(action_input,c_net[0])+c_net[1]),c_net[2])+c_net[3];
        return state_input,action_input,q_value_output,target_update,program_order

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch,program_order):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.program_order:program_order
            })

    def gradients(self,state_batch,action_batch,program_order):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.program_order:program_order
            })[0]

    def target_q(self,state_batch,action_batch,program_order):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch,
            self.target_program_order:program_order
            })

    def q_value(self,state_batch,action_batch,program_order):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.program_order:program_order})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
        
