#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:40:32 2019

@author: chong
"""

import numpy as np
import tensorflow as tf 
import SV_eq_v1 as SV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NN_SV:
    def __init__(self,T,X,tnum,xnum,n,R):
        #parameters of SV system
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        self.n=n
        self.R=R
        
        #parameters of NN (encoding and decoding)
        self.NN_en_n=int(self.xnum*3/4)#number of nodes encoding
        self.NN_de_n=int(self.xnum*3/4)#number of nodes decoding
        self.NN_ed_n=int(self.xnum/3)#number of nodes of encoded layers/ input dimension of EDMD
        print(self.xnum,self.tnum,self.NN_de_n)
        self.bc_size=2
        
        
        #parameter of EDMD
        self.batch_size=1
        self.state_size=20
        self.steps=100000
        self.lr=0.1
    
    #generate data based on SV equations
    def data_generate(self):
        
        def SV_modeldata():
            sv=SV.SV_eq_v1(self.T,self.X,self.tnum,self.xnum,self.n,self.R)
            sv.sim()
            return sv.Q,sv.A,sv.Z
        self.Q,self.A,self.Z=SV_modeldata()
        self.Q,self.A,self.Z=self.Q.T,self.A.T,self.Z.T
        self.xnum,self.tnum=self.Z.shape[1],self.Z.shape[0]
        
    
    
    def _build_model(self):
        #LSTM model
        with tf.variable_scope('foo',reuse=tf.AUTO_REUSE):

            self.bc = tf.placeholder(tf.float32, [self.batch_size,self.tnum,2], name='bc')
            self.Qic=tf.placeholder(tf.float32,[self.batch_size,self.xnum], name='Qic')
            self.Aic=tf.placeholder(tf.float32,[self.batch_size,self.xnum], name='Aic')
            
            self.Qpre = tf.placeholder(tf.float32, [self.batch_size,self.tnum,self.xnum], name='Qout')
            self.Apre = tf.placeholder(tf.float32, [self.batch_size,self.tnum,self.xnum], name='Aout')
            
            #将输入的QA转为[self.batch_size,self.tnum,self.state_size]的东西输入
            with tf.variable_scope('icin'):
                self.WQic = tf.get_variable('WQ0', [self.xnum, self.state_size])
                self.bQic = tf.get_variable('bQ0', [self.state_size], initializer=tf.constant_initializer(0.0))
                self.WAic = tf.get_variable('WA0', [self.xnum, self.state_size])
                self.bAic = tf.get_variable('bA0', [self.state_size], initializer=tf.constant_initializer(0.0))
            
            with tf.variable_scope('Q'):
                self.WQ = tf.get_variable('WQ1', [self.state_size, self.xnum])
                self.bQ = tf.get_variable('bQ1', [self.xnum], initializer=tf.constant_initializer(0.0))
                self.WQ2 = tf.get_variable('WQ2', [self.xnum, self.xnum])
                self.bQ2 = tf.get_variable('bQ2', [self.xnum], initializer=tf.constant_initializer(0.0))
            
            with tf.variable_scope('A'):
                self.WA = tf.get_variable('WA1', [self.state_size, self.xnum])
                self.bA = tf.get_variable('bA1', [self.xnum], initializer=tf.constant_initializer(0.0))
                self.WA2 = tf.get_variable('WQ2', [self.xnum, self.xnum])
                self.bA2 = tf.get_variable('bQ2', [self.xnum], initializer=tf.constant_initializer(0.0))
            
            
            self.init_state = tf.reshape(tf.nn.tanh(tf.matmul(self.Qic,self.WQic)+self.bQic)
                                        +tf.nn.tanh(tf.matmul(self.Aic,self.WAic)+self.bAic)
                                                    ,[self.batch_size, self.state_size])
            #print(self.init_state.shape)
            self.rnn_inputs = self.bc
            #print(self.rnn_inputs.shape)
            #注意这里去掉了这行代码，因为我们不需要将其表示成列表的形式在使用循环去做。
            #rnn_inputs = tf.unstack(x_one_hot, axis=1)
            self.cell = tf.contrib.rnn.BasicRNNCell(self.state_size)
            #使用dynamic_rnn函数，动态构建RNN模型
            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, 
                                                                   self.rnn_inputs, 
                                                                   initial_state=self.init_state)
            
            #print(self.rnn_outputs.shape,self.final_state.shape)
            
            self.Qout=tf.matmul(tf.nn.tanh(tf.matmul(self.rnn_outputs,self.WQ)+self.bQ),self.WQ2)+self.bQ2
            self.Aout=tf.matmul(tf.nn.tanh(tf.matmul(self.rnn_outputs,self.WA)+self.bA),self.WA2)+self.bA2

            self.losses = tf.sqrt(tf.square(self.Qout - self.Qpre)+tf.square(self.Aout - self.Apre))
            self.total_loss = tf.reduce_mean(self.losses)
            self.train = tf.train.AdagradOptimizer(self.lr).minimize(self.total_loss)

    #training
    def training(self):
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        D1=self.A
        D2=self.Q
        print(D1.shape,D2.shape)
        maxu1=np.max(D1)
        minu1=np.min(D1)
        maxu2=np.max(D2)
        minu2=np.min(D2)
        print(maxu1,minu1,maxu2,minu2)
        
        def normalize(u,maxu1,minu1):
            u_norm=[]
            n,m=u.shape
            for i in range(n):
                tem=[]
                for j in range(m):
                    tem.append((u[i,j]-minu1)/(maxu1-minu1))
                u_norm.append(tem)
            
            u_norm=np.array(u_norm)
            return u_norm
        def vnormalize(u_K,maxu1,minu1):
            u_vnorm=[]
            n,m=u_K.shape
            for i in range(n):
                tem=[]
                for j in range(m):
                    tem.append(u_K[i,j]*(maxu1-minu1)+minu1)
                u_vnorm.append(tem)
            
            u_vnorm=np.array(u_vnorm)
            return u_vnorm
        
        D1=normalize(D1,maxu1,minu1)
        D2=normalize(D2,maxu2,minu2)
        Qic=np.array(D1[0])
        Aic=np.array(D2[0])
        bc=np.array([D1[:,0],D2[:,0]])
        Qp=np.array(D1)
        Ap=np.array(D2)
        bc=bc.reshape(1,self.tnum,2)
        Qic=Qic.reshape(1,self.xnum)
        Aic=Aic.reshape(1,self.xnum)
        Qp=Qp.reshape(1,self.tnum,self.xnum)
        Ap=Ap.reshape(1,self.tnum,self.xnum)
        
        er=[]
        for j in range(self.steps):
            if j>int(self.steps*4/10) and j<int(self.steps*8/10):
                self.lr=0.001
            if j>int(self.steps*9/10):
                self.lr=0.0001
            self.sess.run(self.train,feed_dict={self.Qic:Qic,
                                                self.Aic:Aic,
                                                self.bc:bc,
                                                self.Qpre:Qp,
                                                self.Apre:Ap})
            r=self.sess.run(self.total_loss,feed_dict={self.Qic:Qic,
                                                self.Aic:Aic,
                                                self.bc:bc,
                                                self.Qpre:Qp,
                                                self.Apre:Ap})
            er.append(r)
            
        figure = plt.figure()
        plt.plot(er)    
    
        Qpp=self.sess.run(self.Qout,feed_dict={self.Qic:Qic,
                                             self.Aic:Aic,
                                             self.bc:bc})  
        App=self.sess.run(self.Aout,feed_dict={self.Qic:Qic,
                                             self.Aic:Aic,
                                             self.bc:bc})  
        resultQ,resultA=Qpp,App#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
        tv0=np.mat(resultQ*(maxu1-minu1)+minu1)
        tv1=np.mat(resultA*(maxu2-minu2)+minu2)
        
        print(tv0.shape,tv1.shape)
        
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/tv0.shape[1])     
        Y = np.arange(0, self.T, self.T/tv0.shape[0])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, tv0, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
    
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/tv1.shape[1])     
        Y = np.arange(0, self.T, self.T/tv1.shape[0])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, tv1, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
    #test
    def test(self):
        pass
        
if __name__=='__main__':
    T=600.6
    tnum=500
    N=20
    xnum=20
    n=0.01
    R=10
    nn=NN_SV(T,N,tnum,xnum,n,R)
    nn.data_generate()
    nn._build_model()
    nn.training()
    
    

       