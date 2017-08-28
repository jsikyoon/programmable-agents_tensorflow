import tensorflow as tf 
import numpy as np
import math
from detector import Detector
from message_passing import Message_passing
from program import Program

# Hyper Parameters
LEARNING_RATE = 1e-6
TAU = 0.001
BATCH_SIZE = 64
order_num=2

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,obj_num,fea_size,action_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.obj_num=obj_num
        self.fea_size=fea_size
        self.action_dim = action_dim
        self.order_num=order_num;
        # create actor network
        self.state_input,self.action_output,self.net,self.program_order = self.create_network(state_dim,obj_num,fea_size,action_dim)

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net,self.target_program_order = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.global_variables_initializer())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

    def create_network(self,state_dim,obj_num,fea_size,action_dim):
        state_input = tf.placeholder("float",[None,state_dim]);
        program_order=tf.placeholder("float",[None,3]);
        # Detector
        self.detector=Detector(self.sess,self.state_dim,self.obj_num,self.fea_size,state_input,"actor");
        d_params=self.detector.net;
        # Program
        self.program=Program(self.sess,self.state_dim,self.obj_num,self.fea_size,self.detector.Theta,program_order,"actor");
        p=self.program.p;
        # Message Passing
        self.message_passing=Message_passing(self.sess,self.state_dim,self.obj_num,self.fea_size,self.program.p,state_input,"actor");
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
        # get a
        with tf.variable_scope('actor_nets'):
          w=tf.get_variable('w',shape=[self.fea_size,self.action_dim]);
          b=tf.get_variable('b',shape=[self.action_dim]);
          action_output=tf.tanh(tf.matmul(tf.tanh(h),w)+b);
        # param list 
        a_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_nets');
        param_list=d_params+m_params+a_params;
        
        return state_input,action_output,param_list,program_order

    def create_target_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim]);
        program_order=tf.placeholder("float",[None,3]);
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        # params for each net
        d_net=net[:self.detector.params_num];
        m_net=net[self.detector.params_num:(self.detector.params_num+self.message_passing.params_num)];
        a_net=net[(self.detector.params_num+self.message_passing.params_num):];
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
        # get a
        action_output=tf.tanh(tf.matmul(tf.tanh(h),a_net[0])+a_net[1]);
        return state_input,action_output,target_update,target_net,program_order;

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch,program_order):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch,
            self.program_order:program_order
            })

    def actions(self,state_batch,program_order):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch,
            self.program_order:program_order
            })

    def action(self,state,program_order):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state,
            self.program_order:program_order
            })[0]


    def target_actions(self,state_batch,program_order):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_program_order:program_order
            })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

        
