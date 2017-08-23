import tensorflow as tf 
import numpy as np
import math

# Parameter
order_num=10;

class Program:
  def __init__(self,sess,state_dim,fea_size,obj_num,Theta):
    self.sess = sess;
    self.state_dim = state_dim;
    self.fea_size=fea_size;
    self.obj_num=obj_num;
    self.order_num=order_num;
    self.Theta=Theta;
    self.p,self.program_order = self.compile_order();

  def compile_order(self):
    self.Theta=tf.reshape(self.Theta,[-1,self.obj_num,9]);
    self.Theta=tf.transpose(self.Theta,perm=[0,2,1]);
    self.Theta=tf.unstack(self.Theta,9,1);
    p=self.Theta[0];
    program_order=tf.placeholder(tf.int32,[self.order_num,3]);
    program_order2=tf.unstack(program_order,self.order_num,0);
    for i in range(self.order_num):
      program_order2[i]=tf.unstack(program_order2[i],3,0);
    for i in range(self.order_num):
      for k in range(9):
        for l in range(k+1,9):
          # not=1, and=2, or=3
          p=tf.cond(tf.equal(program_order2[i][0],1)&tf.equal(program_order2[i][1],k),lambda:1-self.Theta[k],lambda:p);
          p=tf.cond(tf.equal(program_order2[i][0],1)&tf.equal(program_order2[i][1],-1),lambda:1-p,lambda:p);
          p=tf.cond(tf.equal(program_order2[i][0],2)&tf.equal(program_order2[i][1],k)&tf.equal(program_order2[i][2],l),lambda:tf.multiply(self.Theta[k],self.Theta[l]),lambda:p);
          p=tf.cond(tf.equal(program_order2[i][0],2)&tf.equal(program_order2[i][1],k)&tf.equal(program_order2[i][2],-1),lambda:tf.multiply(self.Theta[k],p),lambda:p);
          p=tf.cond(tf.equal(program_order2[i][0],3)&tf.equal(program_order2[i][1],k)&tf.equal(program_order2[i][2],l),lambda:self.Theta[k]+self.Theta[l]-tf.multiply(self.Theta[k],self.Theta[l]),lambda:p);
          p=tf.cond(tf.equal(program_order2[i][0],3)&tf.equal(program_order2[i][1],k)&tf.equal(program_order2[i][2],l),lambda:self.Theta[k]+p-tf.multiply(self.Theta[k],p),lambda:p);
    return p, program_order;
		
