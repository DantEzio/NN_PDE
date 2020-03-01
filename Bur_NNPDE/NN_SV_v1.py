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
        
        #parameter of EDMD
        self.steps=500
    
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
        self.x_=tf.placeholder(tf.float32,[None,2,self.xnum])
        #self.y_ed=tf.placeholder(tf.float32,[None,2,self.xnum])#en-decoding output_
        self.y_pre=tf.placeholder(tf.float32,[None,2,self.xnum])#Kx output_
        
        #K近似
        self.W1=tf.Variable(tf.truncated_normal([self.xnum,self.NN_de_n],stddev=0.1))
        self.b1=tf.Variable(tf.zeros([self.NN_de_n]))
        self.W2=tf.Variable(tf.truncated_normal([self.NN_de_n,self.xnum],stddev=0.1))
        self.b2=tf.Variable(tf.zeros([self.xnum]))
    
    #EDMD
    def EDMD(self):
        
        self.ht1=tf.nn.tanh(tf.matmul(self.x_,self.W1)+self.b1)
        self.ht2=tf.nn.tanh(tf.matmul(self.ht1,self.W2)+self.b2)
        
        self.loss2=tf.sqrt(tf.reduce_mean(tf.square(self.ht2 - self.y_pre)))
        
        self.train1=tf.train.AdamOptimizer(0.001).minimize(self.loss2)
    
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
            for i in range(D1.shape[0]-1):
                bx=np.array([D1[i],D2[i]])
                #byed=np.array([D1[i],D2[i]])
                bypre=np.array([D1[i+1],D2[i+1]])
                
                bx=bx.reshape(1,bx.shape[0],bx.shape[1])
                #byed=bx.reshape(1,byed.shape[0],byed.shape[1])
                bypre=bx.reshape(1,bypre.shape[0],bypre.shape[1])
                #print(bx.shape,bypre.shape,byed.shape)
                #print(self.x_.shape,self.y_ed.shape,self.y_pre.shape)
                self.sess.run(self.train1,feed_dict={self.x_:bx,self.y_pre:bypre})
                
        output1,output2=[],[]
        
        for i in range(D1.shape[0]):
            batch_xs=np.array([D1[i],D2[i]])
            batch_xs=batch_xs.reshape(1,batch_xs.shape[0],batch_xs.shape[1])
            pre=self.sess.run(self.ht2,feed_dict={self.x_:batch_xs})      
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
        
        
        #print(np.mat(tv0).shape)
        #print(np.mat(tv1).shape)
        
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/self.xnum)     
        Y = np.arange(0, self.T, self.T/(self.tnum))
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, np.mat(tv0), rstride=1, cstride=1, cmap='rainbow')
        plt.show()
    
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/self.xnum)     
        Y = np.arange(0, self.T, self.T/(self.tnum))
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, np.mat(tv1), rstride=1, cstride=1, cmap='rainbow')
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
    nn.EDMD()
    nn.training()
    
    

       