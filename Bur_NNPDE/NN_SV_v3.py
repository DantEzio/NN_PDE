#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:33:53 2019

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
        self.steps=100
        self.lr=0.001
    
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
        with tf.variable_scope('foo',reuse=tf.AUTO_REUSE):

            lr=0.001
            
            self.ic=tf.placeholder(tf.float32,[None,2,self.xnum],name='ic')#state
            self.bc=tf.placeholder(tf.float32,[None,2,1],name='bc')#bc
            self.y_pre=tf.placeholder(tf.float32,[None,2,self.xnum],name='y')#output of each time step
            self.h0=tf.placeholder(tf.float32,[None,2,self.NN_de_n],name='h')
            #cell
            #self.lstm_cell0=tf.nn.rnn_cell.BasicLSTMCell(self.NN_de_n,forget_bias=0.8) #For LSTM
            #lstm_cell0=tf.nn.rnn_cell.BasicRNNCell(h_size) #For RNN
            
            self.Wic=tf.Variable(tf.truncated_normal([self.xnum,self.NN_de_n],stddev=0.1))
            self.bic=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.Wbc=tf.Variable(tf.truncated_normal([1,self.NN_de_n],stddev=0.1))
            self.bbc=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.W=tf.Variable(tf.truncated_normal([self.NN_de_n,self.xnum],stddev=0.1))
            self.b=tf.Variable(tf.zeros([self.xnum]))
            
            self.Ws1=tf.Variable(tf.truncated_normal([self.NN_de_n,self.NN_de_n],stddev=0.1))
            self.bs1=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.Ws2=tf.Variable(tf.truncated_normal([self.NN_de_n,self.NN_de_n],stddev=0.1))
            self.bs2=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.icen=tf.nn.tanh(tf.matmul(self.ic,self.Wic)+self.bic)
            #self.bcen=tf.nn.tanh(tf.matmul(self.bc,self.Wbc)+self.bbc)
            '''
            self.a1=tf.matmul(self.icen[:,0,:],self.Ws1)+self.bs1
            self.a2=tf.matmul(self.icen[:,1,:],self.Ws2)+self.bs2
            
            self.a1=tf.reshape(self.a1,[-1,self.NN_de_n])
            self.a2=tf.reshape(self.a2,[-1,self.NN_de_n])
            
            self.h1=tf.nn.tanh(self.a1
                              +self.a2
                              +tf.matmul(self.bc,self.Wbc)+self.bbc)
            '''
            
            self.h1=tf.nn.tanh(self.h0
                              +tf.matmul(self.icen,self.Ws1)+self.bs1
                              +tf.matmul(self.bc,self.Wbc)+self.bbc)
            
            
            self.o=tf.nn.tanh(tf.matmul(self.h1,self.W)+self.b)
            
            self.loss=tf.square(self.o - self.y_pre)
            self.train=tf.train.AdamOptimizer(lr).minimize(self.loss)
            
    
    #training
    def training(self):
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        D1=self.A
        D2=self.Q
        
        maxu1=np.max(D1)
        minu1=np.min(D1)
        maxu2=np.max(D2)
        minu2=np.min(D2)
        #print(maxu1,minu1,maxu2,minu2)
        
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
        h0=np.zeros((1,2,self.NN_de_n))
        
        for j in range(self.steps):
            for i in range(self.tnum-10):
                ic=np.array([D1[i],D2[i]])
                bc=np.array([D1[i+10,0],D2[i+10,0]])
                ypre=np.array([D1[i+10],D2[i+10]])
                #print(h0.shape)
                ic=ic.reshape(1,2,self.xnum)
                bc=bc.reshape(1,2,1)
                ypre=ypre.reshape(1,2,self.xnum)
                #print(ic.shape,bc.shape,ypre.shape)
                    
                self.sess.run(self.train,feed_dict={self.ic:ic,self.bc:bc,self.y_pre:ypre,self.h0:h0})
                h0=self.sess.run(self.h1,feed_dict={self.ic:ic,self.bc:bc,self.h0:h0,self.y_pre:ypre})
                
                
            
            
            
        output1,output2=[],[]
        
        ic=np.array([D1[0],D2[0]])
        bc=np.array([D1[1,0],D2[1,0]])
        ic=ic.reshape(1,2,self.xnum)
        bc=bc.reshape(1,2,1)
        
        h0=np.zeros((1,2,self.NN_de_n))
        
        for t in range(self.tnum):
            
            result=self.sess.run(self.o,feed_dict={self.ic:ic,
                                                    self.bc:bc,
                                                    self.h0:h0})
            h0=self.sess.run(self.h1,feed_dict={self.ic:ic,self.bc:bc,self.h0:h0})
            output1.append(result[0][0]*(maxu1-minu1)+minu1)
            output2.append(result[0][1]*(maxu2-minu2)+minu2)
            #print(result.shape)
            ic=result
            bc=np.array([D1[t,0],D2[t,0]])
            ic=ic.reshape(1,2,self.xnum)
            bc=bc.reshape(1,2,1)
            
        result0=np.array(output1)
        result1=np.array(output2)
        tv0=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv0.append(result0[i])
        
        tv1=[]
        for i in range(result1.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv1.append(result1[i])
        
        
        
        tv0=np.mat(tv0)
        tv1=np.mat(tv1)
        
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
    
    

       