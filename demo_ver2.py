#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:03:06 2019

@author: chong
"""

"""
Created on Thu Jul 18 14:13:31 2019

@author: chong


Koopman operator and NN4pde compare

dy/dt=sin(t),y=-cos(t)

bc:y(0)=-cos(0)=-1
"""
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Koopman operator based
'''

def Koopman(t,y,steps):
    #steps=100
    
    n,m=y.shape
    
    x_=tf.placeholder(tf.float32,[None,m])
    y_=tf.placeholder(tf.float32,[None,m])
    
    h_size=10
    
    W1=tf.Variable(tf.truncated_normal([m,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W3=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b3=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,m],stddev=0.1))
    b2=tf.Variable(tf.zeros([m]))
    
    h1=tf.nn.relu(tf.matmul(x_,W1)+b1)
    h2=tf.nn.relu(tf.matmul(h1,W3)+b3)
    
    out_=tf.matmul(h2,W2)+b2
    
    loss1=tf.sqrt(tf.reduce_mean(tf.square(out_ - y_)))
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array([y[i]])
                #bx.reshape([1,3])
                by=np.array([y[i+1]])
                #by.reshape([1,m])
                sess.run(train1,feed_dict={x_:bx,y_:by})
                
        output1=[]
        for i in range(n):
            batch_xs=np.array([y[i]])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(out_,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
        
        
        return tv


'''
NN for PDE based
'''
def PDEE(t,y,steps):
    
    h_size=6
    
    nnx_=tf.placeholder(tf.float32,[None,1])
    
    nnW1=tf.Variable(tf.truncated_normal([1,h_size],stddev=0.1))
    nnb1=tf.Variable(tf.zeros([h_size]))
    
    nnW2=tf.Variable(tf.truncated_normal([h_size,1],stddev=0.1))
    nnb2=tf.Variable(tf.zeros([1]))
    
    nnh1=tf.nn.tanh(tf.matmul(nnx_,nnW1)+nnb1)
    nnout_=tf.nn.tanh(tf.matmul(nnh1,nnW2)+nnb2)
    
    #dt=((4*tf.exp(-2*(nnW2*nnh1+nnb2)))/((tf.exp(-2*(nnW2*nnh1+nnb2))+1)**2))*nnW2*nnW1*((4*tf.exp(-2*(nnW1*nnx_+nnb1)))/((tf.exp(-2*(nnW1*nnx_+nnb1))+1)**2))
    grad_t = tf.gradients(nnout_,nnx_)
    #grad_tt=tf.gradients(grad_t,nnx_)
    
    nnloss=tf.square(grad_t-tf.sin(nnx_))
    
    nntrain=tf.train.AdamOptimizer(0.001).minimize(nnloss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(steps):
            for i in range(len(t)):
                bx=np.array([[t[i]]])
                sess.run(nntrain,feed_dict={nnx_:bx})
                
        output1=[]
        for i in range(len(t)):
            batch_xs=np.array([[t[i]]])
            pre=sess.run(nnout_,feed_dict={nnx_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        p=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
            p.append(y[i][0])
        
        #plt.plot(t,tv)
        #plt.plot(t,p)   
        return tv


def o_model_ver1(T,N,tnum,xnum,Qic,Qbc,belt,alph,q):
  
    deltx=N/xnum
    deltt=T/tnum
    
    
    
    Q=[]

    for i in range(tnum):
        Qtm=[]
        if i==0:
            for j in range(xnum):
                Qtm.append(Qic[j])        
        else:
            for j in range(xnum):
                if j==0:
                    Qtm.append(Qbc[i])
                else:
                    Qt=(Q[i-1][j]+((q[i-1][j]-(Q[i-1][j]-Q[i-1][-1])/deltx)*(Q[i-1][j-1])**(1-belt)/(alph*belt))*deltt)
                    Qtm.append(Qt)
        Q.append(Qtm)

    
    return Q
    



if __name__=='__main__':
    
    
    
    T=10
    tnum=50
    N=20
    xnum=100
    
    hic=[]
    hbc=[]
    uic=[]
    ubc=[]
    
    for i in range(tnum):
        ubc.append(math.sin(i*math.pi/xnum))
        hbc.append(math.cos(i*math.pi/xnum))
        
    for j in range(xnum):
        hic.append(0.01)
        uic.append(0.02)
    
    
    q=[]
    for i in range(tnum):
        qtem=[]
        for j in range(xnum):
            qtem.append(0)
        q.append(qtem)
    
    q=np.array(q)
    print(q.shape)
    
    belt=3/5
    n=0.02
    B=5
    S0=0.01
    alph=(n*B**(2/3)/(S0**(-0.5)))**belt
    
    Q=o_model_ver1(T,N,tnum,xnum,uic,ubc,belt,alph,q)    
    Q=np.array(Q)
    
    print(Q.shape)
    plt.plot(Q)
    
    maxu=np.max(Q)
    minu=np.min(Q)
    
    print(maxu,minu)
    
    n,m=Q.shape
    
    
    u_norm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append((Q[i,j]-minu)/(maxu-minu))
        u_norm.append(tem)
    
    u_norm=np.array(u_norm)
    
    #plt.plot(u_norm)
    
        
    deltx=N/xnum
    deltt=T/tnum
    
    t=[]
    for i in range(tnum):
        t.append(i*deltt)
    
    t=np.array(t)
    
    steps=1000
    Q_K=np.array(Koopman(t,Q,steps))
    print(Q_K.shape)
    
    
    Q_vnorm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append(Q_K[i,j]*(maxu-minu)+minu)
        Q_vnorm.append(tem)
    
    Q_vnorm=np.array(Q_vnorm)
    
    plt.plot(Q_K)
    print(np.max(np.abs(Q_K-Q)))
    #PDEE(t,y,steps)
    np.savetxt('u.txt',Q)
    np.savetxt('u_K.txt',Q_K)
    
    