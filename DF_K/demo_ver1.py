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

    def _build_NNu(self):
        self.W1=tf.Variable(tf.truncated_normal([self.u_in,self.h_size],stddev=0.1))
        self.b1=tf.Variable(tf.zeros([self.h_size]))
        self.W11=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11=tf.Variable(tf.zeros([self.h_size]))
        self.W2=tf.Variable(tf.truncated_normal([self.h_size,self.u_in],stddev=0.1))
        self.b2=tf.Variable(tf.zeros([self.u_in]))
        
        self.l1 = tf.nn.tanh(tf.matmul(self.x,self.W1)+self.b1)
        self.l2 = tf.nn.tanh(tf.matmul(self.l1,self.W11)+self.b11)
        self.u = tf.nn.tanh(tf.matmul(self.l2,self.W2)+self.b2)
        
    def _build_Encoding(self):
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

    def _build_Decoding(self):
        self.W1d=tf.Variable(tf.truncated_normal([self.encoding_in,self.h_size],stddev=0.1))
        self.b1d=tf.Variable(tf.zeros([self.h_size]))
        self.W11d=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11d=tf.Variable(tf.zeros([self.h_size]))
        self.W2d=tf.Variable(tf.truncated_normal([self.h_size,self.u_in],stddev=0.1))
        self.b2d=tf.Variable(tf.zeros([self.u_in]))
            
        self.h1d=tf.nn.tanh(tf.matmul(self.encodu,self.W1d)+self.b1d)
        self.h11d=tf.nn.tanh(tf.matmul(self.h1d,self.W11d)+self.b11d)
        self.decodu = tf.nn.tanh(tf.matmul(self.h11d,self.W2d)+self.b2d)
    
    def _build_K(self):
        self.K = tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        self.out_s0=tf.matmul(self.encodu,self.K)
        self.out_s1=tf.nn.tanh(tf.matmul(self.out_s0,self.W1d)+self.b1d)
        self.out_s11=tf.nn.tanh(tf.matmul(self.out_s1,self.W11d)+self.b11d)
        self.out_s2=tf.nn.tanh(tf.matmul(self.out_s11,self.W2d)+self.b2d)
        
    def _build_sim(self):
        self.u_0=tf.placeholder(tf.float32, [None, self.u_in], 'x')
        
        self.sim_h1e=tf.nn.tanh(tf.matmul(self.u_0,self.W1e)+self.b1e)
        self.sim_h11e=tf.nn.tanh(tf.matmul(self.sim_h1e,self.W11e)+self.b11e)
        self.sim_encodu=tf.nn.tanh(tf.matmul(self.sim_h11e,self.W2e)+self.b2e)
        
        self.u_m=tf.matmul(self.sim_encodu,self.K)
        
        self.sim_h1d=tf.nn.tanh(tf.matmul(self.u_m,self.W1d)+self.b1d)
        self.sim_h11d=tf.nn.tanh(tf.matmul(self.sim_h1d,self.W11d)+self.b11d)
        self.u_t = tf.nn.tanh(tf.matmul(self.sim_h11d,self.W2d)+self.b2d)
        
    def _build_PI(self):
        self.er1=(self.out_s2-self.u)/self.deltt
        self.er2=tf.gradients(self.u,self.x)
        self.er3=tf.gradients(self.er2,self.x)
        self.er=self.er1+tf.multiply(self.u,self.er2)-tf.Variable(self.D)*self.er3
        
        self.loss1=tf.reduce_mean(tf.square(self.u-self.decodu))
        self.loss2=tf.reduce_mean(tf.square(self.er))
        
        self.loss=self.loss1+self.loss2
    
        self.opt=tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss)
    
    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.u_in], 'x')

        self._build_NNu()
        self._build_Encoding()
        self._build_Decoding()
        self._build_K()
        self._build_sim()
        self._build_PI()
        
    
    def o_model_ver1(self,T,N,tnum,xnum,uic,ubc,belt):
  
        deltx=N/xnum
        deltt=T/tnum
        
        #r=a*deltt/deltx
        r=deltt/deltx
        u=[]
        
        u.append(uic)
        
        
        for n in range(tnum):
            ut=[]
            ut.append(ubc[n])
            for j in range(1,xnum-1):
                '''
                if testlag==0:
                    temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])+0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])
                else:
                    temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])+0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])+belt*0.5*deltt/(deltx*deltx)*(u[n][j+1]-2*u[n][j]+u[n][j-1]) 
                '''
                temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])+0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])+belt*0.5*deltt/(deltx*deltx)*(u[n][j+1]-2*u[n][j]+u[n][j-1]) 
                ut.append(round(temu,2))
                if j==xnum-2:
                    ut.append(round(temu,2))
            u.append(ut)   
        return u
    
    
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
            T=20
            tnum=100
            N=50
            
            uic=[]
            ubc=[]
            
            for i in range(tnum):
                ubc.append(np.sin(i*np.pi/tnum))
            
            for j in range(self.u_in):
                uic.append(0.0)
            
            
            test_results=[]
            u0=np.array(uic).reshape(1,self.u_in)
            for t in range(tnum):
                u0[0]+=ubc[t]
                if t==0:
                    print(u0)
                um=sess.run(self.u_t,feed_dict={self.u_0:u0})
                test_results.append(um[0])
                u0=um
            
            
            uc=self.o_model_ver1(T,N,tnum,self.u_in,uic,ubc,self.D)
            
            
        return results,test_results,uc
            
            
if __name__=='__main__':
    
    
    
    
    model=DF_K(50,4)
    results,tr,uc=model.train()
    
    df = pd.DataFrame.from_dict(tr)
    df.to_csv('result.csv', index=False, encoding='utf-8')
    
    df = pd.DataFrame.from_dict(uc)
    df.to_csv('resultuc.csv', index=False, encoding='utf-8')
    
    plt.figure(figsize=(20,8))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    
    ax1.plot(results[1:-1])
    ax2.plot(tr)
    ax3.plot(uc)
    
    plt.show()
    
    
    
    