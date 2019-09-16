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
        
        self.D=2.4
        self.deltt=0.2
        self.deltx=1.0
        
        self.ep=1000
        self.h_size=20
        self.batch_size=100
        
        self.model=self._build_model()
        
    def _build_Encoding(self):
        self.W1e=tf.Variable(tf.truncated_normal([self.u_in,self.h_size],stddev=0.1))
        self.b1e=tf.Variable(tf.zeros([self.h_size]))
        self.W11e=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11e=tf.Variable(tf.zeros([self.h_size]))
        self.W2e=tf.Variable(tf.truncated_normal([self.h_size,self.encoding_in],stddev=0.1))
        self.b2e=tf.Variable(tf.zeros([self.encoding_in]))
        
        self.h1e=tf.nn.relu(tf.matmul(self.u_0,self.W1e)+self.b1e)
        self.h11e=tf.nn.relu(tf.matmul(self.h1e,self.W11e)+self.b11e)
        self.encodu=tf.nn.relu(tf.matmul(self.h11e,self.W2e)+self.b2e)
        #K*encodu

    def _build_Decoding(self):
        self.W1d=tf.Variable(tf.truncated_normal([self.encoding_in,self.h_size],stddev=0.1))
        self.b1d=tf.Variable(tf.zeros([self.h_size]))
        self.W11d=tf.Variable(tf.truncated_normal([self.h_size,self.h_size],stddev=0.1))
        self.b11d=tf.Variable(tf.zeros([self.h_size]))
        self.W2d=tf.Variable(tf.truncated_normal([self.h_size,self.u_in],stddev=0.1))
        self.b2d=tf.Variable(tf.zeros([self.u_in]))
            
        self.h1d=tf.nn.relu(tf.matmul(self.encodu,self.W1d)+self.b1d)
        self.h11d=tf.nn.relu(tf.matmul(self.h1d,self.W11d)+self.b11d)
        self.decodu = tf.nn.relu(tf.matmul(self.h11d,self.W2d)+self.b2d)
    
    def _build_K(self):
        self.K = tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        
    def _build_sim(self):
        
        self.sim_h1e=tf.nn.relu(tf.matmul(self.u_0,self.W1e)+self.b1e)
        self.sim_h11e=tf.nn.relu(tf.matmul(self.sim_h1e,self.W11e)+self.b11e)
        self.sim_encodu=tf.nn.relu(tf.matmul(self.sim_h11e,self.W2e)+self.b2e)
        
        self.u_m=tf.matmul(self.sim_encodu,self.K)
        
        self.sim_h1d=tf.nn.relu(tf.matmul(self.u_m,self.W1d)+self.b1d)
        self.sim_h11d=tf.nn.relu(tf.matmul(self.sim_h1d,self.W11d)+self.b11d)
        self.u_t = tf.nn.relu(tf.matmul(self.sim_h11d,self.W2d)+self.b2d)
        
    def _build_PI(self):
        
        #losspre结合PED能量不等式
        tA=[]
        for i in range(self.u_in):
            tem=[]
            if i==0:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(0.0)
                    elif j==i+1:
                        tem.append(-0.5*self.deltt/self.deltx)
                    else:
                        tem.append(0.0)
            
            elif i==self.encoding_in-1:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(0.0)
                    elif j==i-1:
                        tem.append(0.5*self.deltt/self.deltx)
                    else:
                        tem.append(0.0)
            
            else:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(0.5*self.deltt/self.deltx)
                    elif j==i+1:
                        tem.append(0.0)
                    elif j==i+2:
                        tem.append(-0.5*self.deltt/self.deltx)
                    else:
                        tem.append(0.0)
            tA.append(tem)
    
        self.A1=tf.cast(tf.constant(np.mat(tA)),tf.float32)
        
        tA=[]
        for i in range(self.u_in):
            tem=[]
            
            if i==0:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(-self.deltt*self.deltt/(self.deltx*self.deltx))
                    elif j==i+1:
                        tem.append(0.5*self.deltt*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            
            elif i==self.encoding_in-1:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(-self.deltt*self.deltt/(self.deltx*self.deltx))
                    elif j==i-1:
                        tem.append(0.5*self.deltt*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            
            else:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(0.5*self.deltt*self.deltt/(self.deltx*self.deltx))
                    elif j==i+1:
                        tem.append(-self.deltt*self.deltt/(self.deltx*self.deltx))
                    elif j==i+2:
                        tem.append(0.5*self.deltt*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            tA.append(tem)
    
        self.A2=tf.cast(tf.constant(np.mat(tA)),tf.float32)
        
        tA=[]
        for i in range(self.u_in):
            tem=[]
            
            if i==0:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(self.D*self.deltt/(self.deltx*self.deltx))
                    elif j==i+1:
                        tem.append(-0.5*self.D*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            
            elif i==self.u_in-1:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(self.D*self.deltt/(self.deltx*self.deltx))
                    elif j==i-1:
                        tem.append(-0.5*self.D*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            
            else:
                for j in range(self.u_in):
                    if j==i:
                        tem.append(-0.5*self.D*self.deltt/(self.deltx*self.deltx))
                    elif j==i+1:
                        tem.append(self.D*self.deltt/(self.deltx*self.deltx))
                    elif j==i+2:
                        tem.append(-0.5*self.D*self.deltt/(self.deltx*self.deltx))
                    else:
                        tem.append(0.0)
            tA.append(tem)
    
        self.A3=tf.cast(tf.constant(np.mat(tA)),tf.float32)
        
        self.t1=tf.matmul(self.u_0,self.A1)
        self.t2=tf.matmul(self.u_0,self.A2)
        self.t3=tf.multiply(self.u_0,self.u_0)
        
        self.loss2=tf.sqrt(tf.reduce_mean(tf.square(tf.multiply(self.t1,self.u_0)+tf.multiply(self.t2,self.t3)+tf.matmul(self.u_0,self.A3)+self.u_0-self.u_t)))
        
        self.loss1=tf.reduce_mean(tf.square(self.u_0-self.decodu))
        
        self.loss=0.5*self.loss1+self.loss2
    
        self.opt_0=tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss)
        
        self.opt_1=tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss1)
        self.opt_2=tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss2)
      
        
    def _build_model(self):
        self.u_0 = tf.placeholder(tf.float32, [None, self.u_in], 'u0')

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
    
    
    def normalize(self,u,maxu1,minu1):
        u_norm=[]
        n,m=u.shape
        for i in range(n):
            tem=[]
            for j in range(m):
                tem.append((u[i,j]-minu1)/(maxu1-minu1))
            u_norm.append(tem)
        
        u_norm=np.array(u_norm)
        return u_norm
    
    def vnormalize(self,u_K,maxu1,minu1):
        u_vnorm=[]
        n,m=u_K.shape
        for i in range(n):
            tem=[]
            for j in range(m):
                tem.append(u_K[i,j]*(maxu1-minu1)+minu1)
            u_vnorm.append(tem)
        
        u_vnorm=np.array(u_vnorm)
        return u_vnorm

    
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
                        temx.append(float(np.random.random(1)[0]))
                    X.append(temx)
            
                for lit in range(100):
                    sess.run(self.opt_1,feed_dict={self.u_0:X})
                for lit in range(100):
                    sess.run(self.opt_2,feed_dict={self.u_0:X})
                for lit in range(100):
                    sess.run(self.opt_0,feed_dict={self.u_0:X})
                    loss_f=sess.run(self.loss,feed_dict={self.u_0:X})
                    loss_set.append(loss_f)
                #print(np.sum(loss_set))
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
                u0=self.normalize(u0,1.0,0.0)
                um=sess.run(self.u_t,feed_dict={self.u_0:u0})
                um=self.vnormalize(um,1.0,0.0)
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
    
    
    
    