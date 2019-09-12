#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:33:37 2019

@author: chong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DF_K:
    def __init__(self,u_in,encoding_in):
        self.u_in=u_in
        self.encoding_in=encoding_in
        
        self.D=0.24
        self.deltt=0.1
        
        self.ep=100
        self.h_size=20
        self.batch_size=10
        
        self.model=self._build_model()

            
    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.u_in], 'x')

        #self._build_NNu()
        self.W1=tf.Variable(tf.truncated_normal([self.u_in,self.h_size],stddev=0.1))
        self.b1=tf.Variable(tf.zeros([self.h_size]))
        self.W11=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11=tf.Variable(tf.zeros([self.h_size]))
        self.W2=tf.Variable(tf.truncated_normal([self.h_size,self.u_in],stddev=0.1))
        self.b2=tf.Variable(tf.zeros([self.u_in]))
        
        self.l1 = tf.nn.tanh(tf.matmul(self.x,self.W1)+self.b1)
        self.l2 = tf.nn.tanh(tf.matmul(self.l1,self.W11)+self.b11)
        self.u = tf.nn.tanh(tf.matmul(self.l2,self.W2)+self.b2)
        
        #self._build_Encoding()
        self.W1e=tf.Variable(tf.truncated_normal([self.u_in,self.h_size],stddev=0.1))
        self.b1e=tf.Variable(tf.zeros([self.h_size]))
        self.W11e=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11e=tf.Variable(tf.zeros([self.h_size]))
        self.W2e=tf.Variable(tf.truncated_normal([self.h_size,self.encoding_in],stddev=0.1))
        self.b2e=tf.Variable(tf.zeros([self.encoding_in]))
        
        self.h1e=tf.nn.tanh(tf.matmul(self.u,self.W1e)+self.b1e)
        self.h11e=tf.nn.tanh(tf.matmul(self.h1e,self.W11e)+self.b11e)
        self.encodu=tf.nn.tanh(tf.matmul(self.h11e,self.W2e)+self.b2e)
        #K*encodu
            
        #self._build_Decoding()
        self.W1d=tf.Variable(tf.truncated_normal([self.encoding_in,self.h_size],stddev=0.1))
        self.b1d=tf.Variable(tf.zeros([self.h_size]))
        self.W11d=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11d=tf.Variable(tf.zeros([self.h_size]))
        self.W2d=tf.Variable(tf.truncated_normal([self.h_size,self.u_in],stddev=0.1))
        self.b2d=tf.Variable(tf.zeros([self.u_in]))
            
        self.h1d=tf.nn.tanh(tf.matmul(self.encodu,self.W1d)+self.b1d)
        self.h11d=tf.nn.tanh(tf.matmul(self.h1d,self.W11d)+self.b11d)
        self.decodu = tf.nn.tanh(tf.matmul(self.h11d,self.W2d)+self.b2d)
            
        #self._build_K()
        self.K = tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        
        self.out_s0=tf.matmul(self.encodu,self.K)
        self.out_s1=tf.nn.tanh(tf.matmul(self.out_s0,self.W1d)+self.b1d)
        self.out_s11=tf.nn.tanh(tf.matmul(self.out_s1,self.W11d)+self.b11d)
        self.out_s2=tf.nn.tanh(tf.matmul(self.out_s11,self.W2d)+self.b2d)
        
        #self._build_PI()
        self.er1=(self.out_s2-self.u)/self.deltt
        self.er2=tf.gradients(self.u,self.x)
        self.er3=tf.gradients(self.er2,self.x)
        self.er=self.er1+tf.multiply(self.u,self.er2)-tf.Variable(self.D)*self.er3
        
        self.loss1=tf.reduce_mean(tf.square(self.u-self.decodu))
        self.loss2=tf.reduce_mean(tf.square(self.er))
        
        self.loss=self.loss1+self.loss2
    
        self.opt=tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss)
    
    def train(self):
        #self.sess.run(tf.global_variables_initializer())
        #tf.reset_default_graph()
        with tf.Session() as sess:
            results=[]
            sess.run(tf.global_variables_initializer())
            for i in range(self.ep):
                #随机采样x构造输入，代入系统训练
                loss_set=[]
                X=[]
                for j in range(self.batch_size):
                    temx=[]
                    for it in range(self.u_in):
                        temx.append(np.random.random(1)[0])
                    X.append(temx)
            
                
                for lit in range(100):
                    sess.run(self.opt,feed_dict={self.x:X})
                    loss_f=sess.run(self.loss,feed_dict={self.x:X})
                    loss_set.append(loss_f)
                print(np.sum(loss_set))
                results.append(np.sum(loss_set))
            
            #test
            Kop=sess.run(self.K)
            a,b=np.linalg.eig(Kop) 
            print(Kop)
            print(a)
            print(b)
            
        return results,Kop
            
            
if __name__=='__main__':
    
    model=DF_K(50,4)
    results,K=model.train()
    
    plt.figure(figsize=(20,8))
    ax1 = plt.subplot(2,1,1)
    plt.plot(results[1:-1])
    plt.show()
    
    