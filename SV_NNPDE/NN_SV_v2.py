#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:25:53 2019

@author: chong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:30:00 2019

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
        self.steps=10
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
            
            self.ic=tf.placeholder(tf.float32,[None,2,self.xnum])#state
            self.bc1=tf.placeholder(tf.float32,[None,2])#bc
            self.bc2=tf.placeholder(tf.float32,[None,2])#bc
            self.bc3=tf.placeholder(tf.float32,[None,2])#bc
            self.bc4=tf.placeholder(tf.float32,[None,2])#bc
            self.bc5=tf.placeholder(tf.float32,[None,2])#bc
            self.bc6=tf.placeholder(tf.float32,[None,2])#bc
            #self.y_ed=tf.placeholder(tf.float32,[None,2,self.xnum])#en-decoding output_
            self.y_pre1=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
            self.y_pre2=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
            self.y_pre3=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
            self.y_pre4=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
            self.y_pre5=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
            self.y_pre6=tf.placeholder(tf.float32,[None,2,self.xnum])#output of each time step
           
            #cell
            #self.lstm_cell0=tf.nn.rnn_cell.BasicLSTMCell(self.NN_de_n,forget_bias=0.8) #For LSTM
            #lstm_cell0=tf.nn.rnn_cell.BasicRNNCell(h_size) #For RNN
            
            self.Wic=tf.Variable(tf.truncated_normal([self.xnum,self.NN_de_n],stddev=0.1))
            self.bic=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.Wbc=tf.Variable(tf.truncated_normal([2,self.NN_de_n],stddev=0.1))
            self.bbc=tf.Variable(tf.zeros([self.NN_de_n]))
            
            self.W=tf.Variable(tf.truncated_normal([self.NN_de_n,self.xnum],stddev=0.1))
            self.b=tf.Variable(tf.zeros([self.xnum]))
            
            self.Ws=tf.Variable(tf.truncated_normal([self.NN_de_n,self.NN_de_n],stddev=0.1))
            self.bs=tf.Variable(tf.zeros([self.NN_de_n]))
            
            
            self.icen=tf.nn.tanh(tf.matmul(self.ic,self.Wic)+self.bic)
            
            self.h1=tf.nn.tanh(tf.matmul(self.icen,self.Ws)+self.bs+tf.matmul(self.bc1,self.Wbc)+self.bbc)
            self.o1=tf.nn.tanh(tf.matmul(self.h1,self.W)+self.b)
            
            self.h2=tf.nn.tanh(tf.matmul(self.h1,self.Ws)+self.bs+tf.matmul(self.bc2,self.Wbc)+self.bbc)
            self.o2=tf.nn.tanh(tf.matmul(self.h2,self.W)+self.b)
            
            self.h3=tf.nn.tanh(tf.matmul(self.h2,self.Ws)+self.bs+tf.matmul(self.bc3,self.Wbc)+self.bbc)
            self.o3=tf.nn.tanh(tf.matmul(self.h3,self.W)+self.b)
            
            self.h4=tf.nn.tanh(tf.matmul(self.h3,self.Ws)+self.bs+tf.matmul(self.bc3,self.Wbc)+self.bbc)
            self.o4=tf.nn.tanh(tf.matmul(self.h4,self.W)+self.b)
            
            self.h5=tf.nn.tanh(tf.matmul(self.h4,self.Ws)+self.bs+tf.matmul(self.bc3,self.Wbc)+self.bbc)
            self.o5=tf.nn.tanh(tf.matmul(self.h5,self.W)+self.b)
            
            self.h6=tf.nn.tanh(tf.matmul(self.h5,self.Ws)+self.bs+tf.matmul(self.bc3,self.Wbc)+self.bbc)
            self.o6=tf.nn.tanh(tf.matmul(self.h6,self.W)+self.b)
            
            
            self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.o1 - self.y_pre1)
                                            +tf.square(self.o2 - self.y_pre2)
                                            +tf.square(self.o3 - self.y_pre3)
                                            +tf.square(self.o4 - self.y_pre4)
                                            +tf.square(self.o5 - self.y_pre5)
                                            +tf.square(self.o6 - self.y_pre6)))
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
        
        for j in range(self.steps):
            for i in range(D1.shape[0]-7):
                ic=np.array([D1[i],D2[i]])
                bc1=np.array([D1[i+1][0],D2[i+1][0]])
                bc2=np.array([D1[i+2][0],D2[i+2][0]])
                bc3=np.array([D1[i+3][0],D2[i+3][0]])
                bc4=np.array([D1[i+4][0],D2[i+4][0]])
                bc5=np.array([D1[i+5][0],D2[i+5][0]])
                bc6=np.array([D1[i+6][0],D2[i+6][0]])
                
                o1=np.array([D1[i+1],D2[i+1]])
                o2=np.array([D1[i+2],D2[i+2]])
                o3=np.array([D1[i+3],D2[i+3]])
                o4=np.array([D1[i+4],D2[i+4]])
                o5=np.array([D1[i+5],D2[i+5]])
                o6=np.array([D1[i+6],D2[i+6]])
                
                ic=ic.reshape(1,ic.shape[0],ic.shape[1])
                bc1=bc1.reshape(1,2)
                bc2=bc2.reshape(1,2)
                bc3=bc3.reshape(1,2)
                bc4=bc4.reshape(1,2)
                bc5=bc5.reshape(1,2)
                bc6=bc6.reshape(1,2)
                o1=o1.reshape(1,o1.shape[0],o1.shape[1])
                o2=o2.reshape(1,o2.shape[0],o2.shape[1])
                o3=o3.reshape(1,o3.shape[0],o3.shape[1])
                o4=o4.reshape(1,o4.shape[0],o4.shape[1])
                o5=o5.reshape(1,o5.shape[0],o5.shape[1])
                o6=o6.reshape(1,o6.shape[0],o6.shape[1])
                
                #print(ic.shape,bc1.shape,o1.shape)
                
                self.sess.run(self.train,feed_dict={self.ic:ic,
                                                    self.bc1:bc1,
                                                    self.bc2:bc2,
                                                    self.bc3:bc3,
                                                    self.bc4:bc4,
                                                    self.bc5:bc5,
                                                    self.bc6:bc6,
                                                    self.y_pre1:o1,
                                                    self.y_pre2:o2,
                                                    self.y_pre3:o3,
                                                    self.y_pre4:o4,
                                                    self.y_pre5:o5,
                                                    self.y_pre6:o6})
                
        output1,output2=[],[]
        
        for i in range(D1.shape[0]-7):
            ic=np.array([D1[i],D2[i]])
            bc1=np.array([D1[i+1][0],D2[i+1][0]])
            bc2=np.array([D1[i+2][0],D2[i+2][0]])
            bc3=np.array([D1[i+3][0],D2[i+3][0]])
            bc4=np.array([D1[i+4][0],D2[i+4][0]])
            bc5=np.array([D1[i+5][0],D2[i+5][0]])
            bc6=np.array([D1[i+6][0],D2[i+6][0]])
            
            ic=ic.reshape(1,ic.shape[0],ic.shape[1])
            bc1=bc1.reshape(1,2)
            bc2=bc2.reshape(1,2)
            bc3=bc3.reshape(1,2)
            bc4=bc4.reshape(1,2)
            bc5=bc5.reshape(1,2)
            bc6=bc6.reshape(1,2)
            
            pre=self.sess.run(self.o3,feed_dict={self.ic:ic,
                                                    self.bc1:bc1,
                                                    self.bc2:bc2,
                                                    self.bc3:bc3,
                                                    self.bc4:bc4,
                                                    self.bc5:bc5,
                                                    self.bc6:bc6})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            #print(result.shape)
            output1.append(result[0][0]*(maxu1-minu1)+minu1)
            output2.append(result[0][1]*(maxu2-minu2)+minu2)
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
    
    

       