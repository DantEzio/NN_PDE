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
    def __init__(self,u_in):
        self.u_in=u_in
        self.encoding_in=int(u_in/3)
        
        self.D=2.4
        
        self.T=20
        self.tnum=100
        self.N=50
        self.deltt=0.2
        self.deltx=1.0
        
        self.ep=40
        self.h_size=self.encoding_in
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
        self.K=tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        '''
        self.K1 = tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        self.k1 = tf.Variable(tf.random_normal([self.encoding_in]))
        self.K2 = tf.Variable(tf.random_normal([self.encoding_in,self.encoding_in]))
        self.k2 = tf.Variable(tf.random_normal([self.encoding_in]))
        '''
        
    def _build_sim(self):
        
        self.u_m2=tf.matmul(self.encodu,self.K)
        '''
        self.u_m1=tf.nn.relu(tf.matmul(self.encodu,self.K1)+self.k1)
        self.u_m2=tf.nn.relu(tf.matmul(self.u_m1,self.K2)+self.k2)
        '''
        
        self.sim_h1d=tf.nn.relu(tf.matmul(self.u_m2,self.W1d)+self.b1d)
        self.sim_h11d=tf.nn.relu(tf.matmul(self.sim_h1d,self.W11d)+self.b11d)
        self.u_t = tf.nn.relu(tf.matmul(self.sim_h11d,self.W2d)+self.b2d)
    
    def _build_uted(self):
        #ut的encoding和decoding
        self.uth1e=tf.nn.relu(tf.matmul(self.u_s,self.W1e)+self.b1e)
        self.uth11e=tf.nn.relu(tf.matmul(self.uth1e,self.W11e)+self.b11e)
        self.utencodu=tf.nn.relu(tf.matmul(self.uth11e,self.W2e)+self.b2e)
        
        self.uth1d=tf.nn.relu(tf.matmul(self.utencodu,self.W1d)+self.b1d)
        self.uth11d=tf.nn.relu(tf.matmul(self.uth1d,self.W11d)+self.b11d)
        self.utdecodu = tf.nn.relu(tf.matmul(self.uth11d,self.W2d)+self.b2d)
    
    
    def _build_PI_ver1(self):
        
        self.loss1=tf.reduce_mean(tf.square(self.u_s-self.u_t))
        
        self.loss2=tf.reduce_mean(tf.square(self.u_0-self.decodu))
        self.loss3=tf.reduce_mean(tf.square(self.u_s-self.utdecodu))
        
        #self.loss=self.loss1+self.loss2
        self.loss=self.loss1+self.loss2+self.loss3
    
        self.opt_0=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss)
        
        self.opt_1=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss1)
        self.opt_2=tf.compat.v1.train.AdamOptimizer(0.001).minimize(self.loss2)
    
    def _build_model(self):
        self.u_0 = tf.placeholder(tf.float32, [None, self.u_in], 'u0')
        self.u_s=tf.placeholder(tf.float32, [None, self.u_in], 'us')

        self._build_Encoding()
        self._build_Decoding()
        self._build_K()
        self._build_sim()
        self._build_uted()
        self._build_PI_ver1()
        
    
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
    
    
    def step(self,uic,T):
        ut=[]
        r=self.deltt/self.deltx
        
        for it in range(T):
            utt=[]
            for j in range(1,self.u_in-1):
                temu=uic[it][j]-0.5*r*uic[it][j]*(uic[it][j+1]-uic[it][j-1])+0.5*r*r*uic[it][j]*uic[it][j]*(uic[it][j+1]-2*uic[it][j]+uic[it][j-1])+self.D*0.5*self.deltt/(self.deltx*self.deltx)*(uic[it][j+1]-2*uic[it][j]+uic[it][j-1]) 
                utt.append(round(temu,2))
                if j==self.u_in-2:
                    utt.append(round(temu,2))
            utt.append(utt[-1])
            ut.append(utt)
        return ut
        
    
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
            results_ed=[]
            results_K=[]
            sess.run(tf.global_variables_initializer())
            
            #data
            uic,ubc=[],[]
            for i in range(self.tnum):
                ubc.append(np.sin(i*np.pi/self.tnum))
            
            for j in range(self.u_in):
                uic.append(0.0)
            uc=self.o_model_ver1(self.T,self.N,self.tnum-1,self.u_in,uic,ubc,self.D)
            X=uc
            Xt=self.step(uc,self.tnum)
            for i in range(self.ep):
                #随机采样x构造输入，代入系统训练
                '''
                X=[]
                for j in range(self.batch_size):
                    temx=[]
                    for it in range(self.u_in):
                        temx.append(float(np.random.random(1)[0]))
                    X.append(temx)
                
                Xt=self.step(X,self.batch_size)
                '''
                for lit in range(100):
   
                    sess.run(self.opt_0,feed_dict={self.u_0:X,self.u_s:Xt})
                    
                loss_f1=sess.run(self.loss2,feed_dict={self.u_0:X,self.u_s:Xt})
                results_ed.append(loss_f1)
                loss_f2=sess.run(self.loss1,feed_dict={self.u_0:X,self.u_s:Xt})
                results_K.append(loss_f2)
            
            #test            
            test_results=sess.run(self.u_t,feed_dict={self.u_0:X})
            test_loss=sess.run(self.loss,feed_dict={self.u_0:X,self.u_s:Xt}) 
            
        return results_ed,results_K,test_loss,test_results,uc
            
            
if __name__=='__main__':
    
    
    
    
    model=DF_K(50)
    r1,r2,test_loss,test_results,uc=model.train()
    '''
    df = pd.DataFrame.from_dict(test_results)
    df.to_csv('result.csv', index=False, encoding='utf-8')
    
    df = pd.DataFrame.from_dict(uc)
    df.to_csv('resultuc.csv', index=False, encoding='utf-8')
    '''
    plt.figure(figsize=(15,8))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    
    uc=np.array(uc).reshape(50,100)
    
    ax1.plot(np.add(r2,r1))
    ax2.plot(uc)
    ax3.plot(test_results)
    
    plt.show()
    
    
    
    