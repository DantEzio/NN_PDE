#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:45:36 2019

@author: chong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SV_DHPM:
    
    def __init__(self):
        #n:神经网络广度；m：神经网络深度
        self.Q_size=10
        self.h_size=10
        
        self._inx=1
        self._int=1
        self._out=1
        self.ep=100
        
        #河宽度
        self.b=3
        self.g=10
        self.KK=0.04
        self.S0=0.02
        
        
    def _build_Q(self):
        self.W1Qx=tf.Variable(tf.truncated_normal([self._inx,self.Q_size],stddev=0.1))
        self.b1Qx=tf.Variable(tf.zeros([self.Q_size]))
        self.W1Qt=tf.Variable(tf.truncated_normal([self._int,self.Q_size],stddev=0.1))
        self.b1Qt=tf.Variable(tf.zeros([self.Q_size]))
        self.W2Q=tf.Variable(tf.truncated_normal([self.Q_size,self.Q_size],stddev=0.1))
        self.b2Q=tf.Variable(tf.zeros([self.Q_size]))
        self.W3Q=tf.Variable(tf.truncated_normal([self.Q_size,self._out],stddev=0.1))
        self.b3Q=tf.Variable(tf.zeros([self._out]))
        
        self.h1Q=tf.nn.relu(tf.matmul(self.x,self.W1Qx)+self.b1Qx+tf.matmul(self.t,self.W1Qt)+self.b1Qt)
        self.h2Q=tf.nn.relu(tf.matmul(self.h1Q,self.W2Q)+self.b2Q)
        self.Q=tf.nn.relu(tf.matmul(self.h2Q,self.W3Q)+self.b3Q)
        
        self.h1Q_sd=tf.nn.relu(tf.matmul(self.x_sd,self.W1Qx)+self.b1Qx+tf.matmul(self.t_sd,self.W1Qt)+self.b1Qt)
        self.h2Q_sd=tf.nn.relu(tf.matmul(self.h1Q_sd,self.W2Q)+self.b2Q)
        self.Q_sd=tf.nn.relu(tf.matmul(self.h2Q_sd,self.W3Q)+self.b3Q)
    
    def _build_h(self):
        self.W1hx=tf.Variable(tf.truncated_normal([self._inx,self.h_size],stddev=0.1))
        self.b1hx=tf.Variable(tf.zeros([self.h_size]))
        self.W1ht=tf.Variable(tf.truncated_normal([self._int,self.h_size],stddev=0.1))
        self.b1ht=tf.Variable(tf.zeros([self.h_size]))
        self.W2h=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b2h=tf.Variable(tf.zeros([self.h_size]))
        self.W3h=tf.Variable(tf.truncated_normal([self.h_size,self._out],stddev=0.1))
        self.b3h=tf.Variable(tf.zeros([self._out]))
        
        self.h1h=tf.nn.relu(tf.matmul(self.x,self.W1hx)+self.b1hx+tf.matmul(self.t,self.W1ht)+self.b1ht)
        self.h2h=tf.nn.relu(tf.matmul(self.h1h,self.W2h)+self.b2h)
        self.h=tf.nn.relu(tf.matmul(self.h2h,self.W3h)+self.b3h)
    
        self.h1h_sd=tf.nn.relu(tf.matmul(self.x_sd,self.W1hx)+self.b1hx+tf.matmul(self.t_sd,self.W1ht)+self.b1ht)
        self.h2h_sd=tf.nn.relu(tf.matmul(self.h1h_sd,self.W2h)+self.b2h)
        self.h_sd=tf.nn.relu(tf.matmul(self.h2h_sd,self.W3h)+self.b3h)
    
    def _build_Qt(self):
        self.Qt = tf.gradients(self.Q,self.t)
               
    def _build_Qx(self):
        self.Qx = tf.gradients(self.Q,self.x)
      
    def _build_ht(self):
        self.ht = tf.gradients(self.h,self.t)
    
    def _build_hx(self):
        self.hx = tf.gradients(self.h,self.x)
        
    def _build_Q2bh(self):
        self.Q2bh=self.Q*self.Q/(self.b*self.h+0.0001)
    
    def _build_Q2bhx(self):
        self.Q2bhx=tf.gradients(self.Q2bh,self.x)
        
    def _build_Qq(self):
        self.Qq=self.Q*tf.abs(self.Q)
    
    
    def _build_PI(self):
        self.I=self.h*self.hx[0]
        self.II=self.h*self.Qq

        self.er_momentum=tf.abs(self.Qt[0]\
                        +self.Q2bhx[0]\
                        +self.g*self.b*self.I\
                        +(self.g/self.KK)*self.b*self.II\
                        -self.g*self.b*self.S0*self.h)
                        
        self.er_mass=tf.abs(self.b*self.ht[0]+self.Qx[0])
        
        #print(self.er_momentum)
        #print(self.er_mass)
        
    
    def _build_SD(self):
        self.er_sd_Q=tf.abs(self.Q_sd-self.sample_Q)
        self.er_sd_h=tf.abs(self.h_sd-self.sample_h)
    
    
    
    def _build_opt(self):
        self.loss_PI=tf.reduce_mean(tf.square(self.er_momentum+self.er_mass))
        self.loss_SD=tf.reduce_mean(tf.square(self.er_sd_Q+self.er_sd_h))
        self.loss=self.loss_PI+self.loss_SD
        
        self.opt_PI=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss_PI)
        self.opt_SD=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss_SD)
        self.opt_PI_SD=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss)
    
    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self._inx], 'x')
        self.t = tf.placeholder(tf.float32, [None, self._int], 't')
        
        self.x_sd = tf.placeholder(tf.float32, [None, self._inx], 'x_sd')
        self.t_sd = tf.placeholder(tf.float32, [None, self._int], 't_sd')
        self.sample_Q=tf.placeholder(tf.float32, [None, self._out], 'sQ')
        self.sample_h=tf.placeholder(tf.float32, [None, self._out], 'sh')
        
        self._build_Q()
        self._build_Qt()
        self._build_Qx()
        self._build_h()
        self._build_ht()
        self._build_hx()
        self._build_Q2bh()
        self._build_Q2bhx()
        self._build_Qq()
        
        self._build_PI()
        self._build_SD()
        self._build_opt()
        

    def train(self,X,T,SDQ,SDh):
        
        er1,er2,er3=[],[],[]
        with tf.Session() as sess:    
            sess.run(tf.global_variables_initializer())
            for i in range(self.ep):
                sess.run(self.opt_PI,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
                sess.run(self.opt_SD,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
                sess.run(self.opt_PI_SD,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})

                er_PI=sess.run(self.loss_PI,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
                er_SD=sess.run(self.loss_SD,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
                er_to=sess.run(self.loss,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
               
                er1.append(er_PI)
                er2.append(er_SD)
                er3.append(er_to)
                
            #test
            Q=sess.run(self.Q,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
            h=sess.run(self.h,feed_dict={self.x:X,self.t:T,self.x_sd:X,self.t_sd:T,self.sample_Q:SDQ,self.sample_h:SDh})
      
        return er1,er2,er3,Q,h
                    
    

if __name__=='__main__':
    sv=SV_DHPM()
    sv._build_model()
    
    #t在0，1之间，x在0，1之间
    
    
    X=np.random.random(100)
    T=np.random.random(100)
    
    SDh,SDQ=[],[]
    for i in range(100):
        SDQ.append(np.sin(i*np.pi/100))
            
    for j in range(100):
        SDh.append(np.cos(j*np.pi/100))
        
    X=np.array(X).reshape(100,1)
    T=np.array(T).reshape(100,1)
    SDQ=np.array(SDQ).reshape(100,1)
    SDh=np.array(SDh).reshape(100,1)
    
    r1,r2,r3,Q,h=sv.train(X,T,SDQ,SDh)
    
    plt.figure(figsize=(15,8))
    ax1 = plt.subplot(7,1,1)
    ax2 = plt.subplot(7,1,2)
    ax3 = plt.subplot(7,1,3)
    ax4 = plt.subplot(7,1,4)
    ax5 = plt.subplot(7,1,5)
    ax6 = plt.subplot(7,1,6)
    ax7 = plt.subplot(7,1,7)
    
    ax1.plot(r1)
    ax2.plot(r2)
    ax3.plot(r3)
    ax4.plot(Q)
    ax5.plot(SDQ)
    ax6.plot(h)
    ax7.plot(SDh)
    
    plt.show()
