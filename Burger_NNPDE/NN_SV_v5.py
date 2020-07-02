# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:38:18 2019

@author: Administrator
说明：该代码使用RNN求解SV方程，RNN初始状态为ic，每个时间节点输入bc[i]，每个时间节点输出Q[i],A[i]
数据来源于SV方法的混合差分公式的多组bcic对应的解，
测试使用额外的icbc
"""

import numpy as np
import tensorflow as tf 
import SV_eq_v2 as SV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NN_SV:
    def __init__(self,T,X,tnum,xnum,n,R):
        #parameters of SV system
        self.T=T
        self.X=X
        self.xnum=xnum*2-2#神经网络的输入数
        self.xn=xnum#模型的空间划分数
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        self.n=n
        self.R=R
        self.rate=0.0
        
        #parameters of NN (encoding and decoding)
        self.NN_en_n=int(self.xnum*3/4)#number of nodes encoding
        self.NN_de_n=int(self.xnum*3/4)#number of nodes decoding
        self.NN_ed_n=int(self.xnum/3)#number of nodes of encoded layers/ input dimension of EDMD
        print(self.xnum,self.tnum,self.NN_de_n)
        self.bc_size=2
        
        
        #parameter of EDMD
        self.batch_size=1
        self.state_size=20
        self.steps=100
        self.lr=0.00012
    
    #generate data based on SV equations
    def data_generate(self):
        
        def SV_modeldata():
            sv=SV.SV_eq_v2(self.T,self.X,self.tnum,self.xn,self.n,self.R,self.rate)
            sv.sim()
            return sv.Q,sv.A,sv.Z
        self.Q,self.A,self.Z=SV_modeldata()
        self.Q,self.A,self.Z=self.Q.T,self.A.T,self.Z.T
        self.xnum,self.tnum=self.Z.shape[1],self.Z.shape[0]
        
    
    def _build_model(self):
        #LSTM model
        with tf.variable_scope('foo',reuse=tf.AUTO_REUSE):

            self.bc = tf.placeholder(tf.float32, [None,self.tnum,2], name='bc')
            self.Qic=tf.placeholder(tf.float32,[None,self.xnum], name='Qic')
            self.Aic=tf.placeholder(tf.float32,[None,self.xnum], name='Aic')
            
            self.Qpre = tf.placeholder(tf.float32, [None,self.tnum,self.xnum], name='Qout')
            self.Apre = tf.placeholder(tf.float32, [None,self.tnum,self.xnum], name='Aout')
            
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
                                                    ,[-1, self.state_size])
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
    
    def get_data(self,A,Q):
        
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
        D1=A
        D2=Q
        maxu1=np.max(D1)
        minu1=np.min(D1)
        maxu2=np.max(D2)
        minu2=np.min(D2)
        #print(maxu1,minu1,maxu2,minu2)
        D1=normalize(D1,maxu1,minu1)
        D2=normalize(D2,maxu2,minu2)
        Qic=np.array(D1[0])
        Aic=np.array(D2[0])
        bc=np.array([D1[:,0],D2[:,0]])
        Qp=np.array(D1)
        Ap=np.array(D2)
        bc=bc.reshape(self.tnum,2)
        Qic=Qic.reshape(self.xnum)
        Aic=Aic.reshape(self.xnum)
        Qp=Qp.reshape(self.tnum,self.xnum)
        Ap=Ap.reshape(self.tnum,self.xnum)
        return bc,Qic,Aic,Qp,Ap,maxu1,minu1,maxu2,minu2

    
    #generate different data based on different bc&ic for training
    def training(self):
        self.sess=tf.Session()
        saver = tf.train.Saver()
        #self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v3_1.ckpt")
        bc,Qic,Aic,Qp,Ap=[],[],[],[],[]
        for i in range(2,3):
            self.rate=i/10
            self.data_generate()
            #print(self.xnum,self.xn)
            bct,Qict,Aict,Qpt,Apt,_,_,_,_=self.get_data(self.A,self.Q) 
            bc.append(bct)
            Qic.append(Qict)
            Aic.append(Aict)
            Qp.append(Qpt)
            Ap.append(Apt)
        
        er=[]
        for j in range(self.steps):
            
            if j>int(self.steps*2/10) and j<=int(self.steps*4/10):
                self.lr=0.0005
            if j>int(self.steps*9/10):
                self.lr=0.0002
            
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
            saver = tf.train.Saver()
            saver_path = saver.save(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v3_1.ckpt")
            print (j,"Model saved in file: ", saver_path,'error:',r)       
        plt.figure()
        plt.plot(er)    
        
        #save model
        saver = tf.train.Saver()
        saver_path = saver.save(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v3_1.ckpt")
        print ("Model saved in file: ", saver_path)
        
    #test
    def test(self):
        #test on new dataset 
        saver = tf.train.Saver()
        self.rate=0.3
        self.data_generate()
        bc,Qic,Aic,Qp,Ap,maxu1,minu1,maxu2,minu2=self.get_data(self.A,self.Q) 
        bc,Qic,Aic,Qp,Ap=[bc],[Qic],[Aic],[Qp],[Ap]
        
        self.sess=tf.Session()
        saver.restore(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v2_18430.ckpt")
        Qpp=self.sess.run(self.Qout,feed_dict={self.Qic:Qic,
                                             self.Aic:Aic,
                                             self.bc:bc})  
        App=self.sess.run(self.Aout,feed_dict={self.Qic:Qic,
                                             self.Aic:Aic,
                                             self.bc:bc})
        r=self.sess.run(self.total_loss,feed_dict={self.Qic:Qic,
                                                self.Aic:Aic,
                                                self.bc:bc,
                                                self.Qpre:Qp,
                                                self.Apre:Ap})
        print(r)
        resultQ,resultA=Qpp,App#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
        tv0=np.mat(resultQ*(maxu1-minu1)+minu1)
        tv1=np.mat(resultA*(maxu2-minu2)+minu2)
        
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
        
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/self.A.shape[1])     
        Y = np.arange(0, self.T, self.T/self.A.shape[0])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, self.A, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/self.Q.shape[1])     
        Y = np.arange(0, self.T, self.T/self.Q.shape[0])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, self.Q, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
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
    #nn.training()
    nn.test()
    

       