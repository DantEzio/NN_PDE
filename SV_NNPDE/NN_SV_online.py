# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:41:42 2020

@author: Administrator
"""


import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc

#计算核
class SV_eq_v2:
    def __init__(self,T,X,tnum,xnum,n,R,rate):
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.n=n
        self.R=R
        
        self.hic,self.hbc,self.Qic,self.Qbc=self.IcBc(rate)
        
    def IcBc(self,rate):
        #根据rate生成不同的bc&ic
        #rate是出现峰的位置
        Qic=[]
        Qbc=[]
        hic=[]
        hbc=[]
        for i in range(self.tnum):
            
            if i >= self.tnum*rate and i<=self.tnum*(0.2+rate):
                Qbc.append(5.0)
            else:
                Qbc.append(0.0)
            
            #Qbc.append(10*np.abs(np.sin(2*np.pi*i/self.tnum)))
            
            hbc.append(1.0)
        
        for j in range(self.xnum*2+1):
            if j<=(self.xnum*2+1)/3:
                Qic.append(0.0)
            else:
                Qic.append(0.0)
            hic.append(1.0)

        return np.array(hic),np.array(hbc),np.array(Qic),np.array(Qbc)
    
    def sim(self):
        
        A=np.ones((self.xnum*2+1,self.tnum))
        Q=np.ones((self.xnum*2+1,self.tnum))
        u=np.ones((self.xnum*2+1,self.tnum))
        Z=np.ones((self.xnum*2+1,self.tnum))
        b=np.ones((self.xnum*2+1,1))
        
        Q[:,0]=self.Qic
        Q[:,1]=self.Qic
        Q[0,:]=self.Qbc
        Z[:,0]=self.hic
        Z[:,1]=self.hic
        Z[0,:]=self.hbc
        
        #print(A.shape,Q.shape)
        
        A[:,0]=Z[:,0]*b.T
        A[:,1]=Z[:,1]*b.T
        A[0,:]=Z[0,:]*b[0]
        
        u[:,0]=Q[:,0]/A[:,0]
        u[:,1]=Q[:,1]/A[:,1]
        u[0,:]=Q[0,:]/A[0,:]
        
        for t in range(2,self.tnum-1):
            
            for i in range(2,self.xnum*2-2):
                
                self.deltt=0.05*self.deltx/(np.max(np.abs(u[:,t]))+np.sqrt(10*np.max(A[:,t]/b[i])))
                
                
                #print(t,i)
                ###############################################################
                
                w=5/8
                v=1/4
                SL_0=w*u[i-2,t]\
                    +(1-w)*u[i,t] \
                    -v*np.sqrt(10*A[i-2,t]/b[i-2])\
                    -(1-v)*np.sqrt(10*A[i,t]/b[i])
                SR_0=(1-w)*u[i-2,t]\
                    +w*u[i,t]\
                    +(1-v)*np.sqrt(10*A[i-2,t]/b[i-2])\
                    +v*np.sqrt(10*A[i,t]/b[i])
                Q2_0=0.0
                if SL_0>0:
                    Q2_0=Q[i-2,t]
                elif SL_0<0 and SR_0>0:
                    Q2_0=(SR_0*Q[i-2,t]-SL_0*Q[i,t]+SL_0*SR_0*(A[i,t]-A[i-2,t]))/(SR_0-SL_0)
                else:
                    Q2_0=Q[i,t]
                    
                SL_1=w*u[i,t]\
                    +(1-w)*u[i+2,t] \
                    -v*np.sqrt(10*A[i,t]/b[i])\
                    -(1-v)*np.sqrt(10*A[i+2,t]/b[i+2])
                SR_1=(1-w)*u[i,t]\
                    +w*u[i+2,t]\
                    +(1-v)*np.sqrt(10*A[i,t]/b[i])\
                    +v*np.sqrt(10*A[i+2,t]/b[i+2])
                Q2_1=0.0
                if SL_1>0:
                    Q2_1=Q[i,t]
                elif SL_1<0 and SR_1>0:
                    Q2_1=(SR_1*Q[i,t]-SL_1*Q[i+2,t]+SL_1*SR_1*(A[i+2,t]-A[i,t]))/(SR_1-SL_1)
                else:
                    Q2_1=Q[i+2,t]
                
                '''
                us0=0.5*(u[i-1,t]+u[i,t])+np.sqrt(10*A[i-1,t]/b[i-1])-np.sqrt(10*A[i,t]/b[i])
                cs0=0.5*(np.sqrt(10*A[i-1,t]/b[i-1])+np.sqrt(10*A[i,t]/b[i]))+0.25*(u[i-1,t]-u[i,t])
                SL_0=np.min([(u[i-1,t]-np.sqrt(10*A[i-1,t]/b[i-1])),us0-cs0])
                SR_0=np.max([(u[i,t]+np.sqrt(10*A[i,t]/b[i])),us0+cs0])
                Q2_0=0.0
                if SL_0>0:
                    Q2_0=Q[i-1,t]
                elif SL_0<0 and SR_0>0:
                    Q2_0=(SR_0*Q[i-1,t]-SL_0*Q[i,t]+SL_0*SR_0*(A[i,t]-A[i-1,t]))/(SR_0-SL_0)
                else:
                    Q2_0=Q[i,t]
                    
                us1=0.5*(u[i,t]+u[i+1,t])+np.sqrt(10*A[i,t]/b[i])-np.sqrt(10*A[i+1,t]/b[i+1])
                cs1=0.5*(np.sqrt(10*A[i,t]/b[i])+np.sqrt(10*A[i+1,t]/b[i+1]))+0.25*(u[i,t]-u[i+1,t])
                SL_1=np.min([(u[i,t]-np.sqrt(10*A[i,t]/b[i])),us1-cs1])
                SR_1=np.max([(u[i+1,t]+np.sqrt(10*A[i+1,t]/b[i+1])),us1+cs1])
                Q2_1=0.0
                if SL_1>0:
                    Q2_1=Q[i,t]
                elif SL_1<0 and SR_1>0:
                    Q2_1=(SR_1*Q[i,t]-SL_1*Q[i+1,t]+SL_1*SR_1*(A[i+1,t]-A[i,t]))/(SR_1-SL_1)
                else:
                    Q2_1=Q[i+1,t]
                '''
                
                #print('Q2:',Q2_0,'Q1:',Q2_1)
                #print(SL_0,SR_0,SL_0,SR_1,t,i)
                ###############################################################
                
                A[i,t+1]=A[i,t]+0.5*(self.deltt/self.deltx)*(Q2_0-Q2_1)#+self.deltt*Si
                #print('Q2-Q1:',Q2_0-Q2_1)
                #print('gama:',self.deltt/self.deltx)
                #print('A:',A[i,t],'At:',A[i,t+1])
                
                if(A[i,t+1]<=0):
                    #print('wrong',t,i)
                    A[i,t+1]=0.01#np.abs(A[i,t+1])
                ###############################################################
                II=0.0
                if Q[i,t]>0 or Q[i,t]==0:
                    II=((Q[i,t]/A[i,t])**2-0.5*(Q[i-2,t]/A[i-2,t])**2)/self.deltx
                else:
                    II=((Q[i+2,t]/A[i+2,t])**2-0.5*(Q[i,t]/A[i,t])**2)/self.deltx
                #print('ii:',II)
                
                
                III=0.0
                if Q[i,t]>0:
                    if i<2:
                        III=(-7*Z[i-1,t]+3*Z[i,t]+3*Z[i+1,t])/(4*self.deltx)
                    else:
                        III=(Z[i-2,t]-7*Z[i-1,t]+3*Z[i,t]+3*Z[i+1,t])/(4*self.deltx)
                elif Q[i,t]==0:
                    III=(Z[i+1,t]-Z[i-1,t])/(self.deltx)
                else:
                    if i<2:
                        III=(7*Z[i+1,t]-3*Z[i,t]-3*Z[i-1,t])/(4*self.deltx)
                    else:
                        III=(-Z[i-2,t]+7*Z[i+1,t]-3*Z[i,t]-3*Z[i-1,t])/(4*self.deltx)
                
                III=0.0
                k=0
                if Q[i,t]>0.0 and Q[i-2,t]>0.0:
                    k=0
                elif Q[i,t]<0.0 and Q[i+2,t]<0.0:
                    k=2
                Cup=(0.25*self.deltt/self.deltx)*(np.abs(u[i+k,t])+np.abs(u[i-2+k,t]))
                Cdown=(0.25*self.deltt/self.deltx)*(np.abs(u[i+2-k,t])+np.abs(u[i-k,t]))
                
                deltZup=(Z[i+k,t]-Z[i-2+k,t])/(2*self.deltx)
                deltZdown=(Z[i+2-k,t]-Z[i-k,t])/(2*self.deltx)
                
                III=np.sqrt(Cup)*deltZup+(1-np.sqrt(Cdown))*deltZdown
                
                
                #III=(Z[i+1,t]-Z[i-1,t])/(4*self.deltx)
                
                #print('iii:',III)
                ###############################################################
                a=Q[i,t]-self.deltt*II-10*A[i,t]*III*self.deltt
                am=1+((10*self.n**2*np.abs(Q[i,t])*self.deltt)/(np.power(self.R,4/3)*A[i,t]))
                Q[i,t+1]=a/am
                
                Z[i,t+1]=A[i,t+1]/b[i]
                u[i,t+1]=Q[i,t+1]/A[i,t+1]
                
            A[A.shape[0]-1,t+1]=A[A.shape[0]-3,t+1]
            Q[Q.shape[0]-1,t+1]=Q[Q.shape[0]-3,t+1]
            Z[Z.shape[0]-1,t+1]=Z[Z.shape[0]-3,t+1]
            u[u.shape[0]-1,t+1]=u[u.shape[0]-3,t+1]
            
            A[A.shape[0]-2,t+1]=A[A.shape[0]-3,t+1]
            Q[Q.shape[0]-2,t+1]=Q[Q.shape[0]-3,t+1]
            Z[Z.shape[0]-2,t+1]=Z[Z.shape[0]-3,t+1]
            u[u.shape[0]-2,t+1]=u[u.shape[0]-3,t+1]
        
        
        #print(Z.shape,Q.shape)
        self.A=A
        self.Q=Q
        self.Z=Z
        


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
        #print(self.xnum,self.tnum,self.NN_de_n)
        self.bc_size=2
        
        
        #parameter of EDMD
        self.batch_size=1
        self.state_size=20
        self.steps=200
        self.lr=0.0009
    
    #generate data based on SV equations
    def data_generate(self):
        
        def SV_modeldata():
            sv=SV_eq_v2(self.T,self.X,self.tnum,self.xn,self.n,self.R,self.rate)
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
        saver.restore(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v2_98.ckpt")
        bc,Qic,Aic,Qp,Ap=[],[],[],[],[]
        for i in range(3,4):
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
            '''
            if j>int(self.steps*2/10) and j<=int(self.steps*4/10):
                self.lr=0.0005
            if j>int(self.steps*9/10):
                self.lr=0.0002
            '''
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
            saver_path = saver.save(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v2_99.ckpt")
            print (j,"Model saved in file: ", saver_path,'error:',r)       
        #plt.figure()
        #plt.plot(er)    
        
        #save model
        saver = tf.train.Saver()
        saver_path = saver.save(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v2_99.ckpt")
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
        saver.restore(self.sess, "D:/Chong/NN_PDE/SV_NNPDE/save/model_v2_3.ckpt")
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
    
    for it in range(1):
        print('training round:',it)
        nn=NN_SV(T,N,tnum,xnum,n,R)
        nn.data_generate()
        nn._build_model()
        nn.training()
        del nn
        gc.collect()
    '''
    nn=NN_SV(T,N,tnum,xnum,n,R)
    nn.data_generate()
    nn._build_model()
    nn.test()
    '''

       