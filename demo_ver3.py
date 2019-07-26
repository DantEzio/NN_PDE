#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:08:45 2019

@author: chong
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


def o_model_ver1(T,N,tnum,xnum,hic,hbc,uic,ubc,i0,c,q):
  
    deltx=N/xnum
    deltt=T/tnum
    g=10
    
    h=[]
    u=[]
    
    h.append(hic)
    u.append(uic)
    
    
    for j in range(tnum):
        ht=[]
        ut=[]
        ht.append(hbc[j])
        ut.append(ubc[j])
        for i in range(xnum-1):

            A=np.mat([[(1/deltt+u[j][i]/deltx),(h[j][i]/deltx)],[g/deltx,(1/deltt+u[j][i]/deltx)]])#2*2
            B=np.mat([[-(u[j][i]/deltx),-(h[j][i]/deltx)],[-(g/deltx),-(u[j][i]/deltx)]])#2*2
            C=np.mat([[-(h[j][i+1]/deltt)-(q[j][i])],[-(u[j][i+1]/deltt)-i0-((u[j][i]**2)/(c**2*h[j][i]))]])#2*1
            f_i_j1=np.mat([[ht[i]],[ut[i]]])#2*1
            
            #f_i1_j1=np.mat()#2*1
            f_i1_j1=A.I*(-B*f_i_j1-C)
            ht.append(float(f_i1_j1[0]))
            ut.append(float(f_i1_j1[1]))
        h.append(ht)
        u.append(ut)   
    return h,u
    



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
    
    i0=0.05
    c=0.01
    
    h,u=o_model_ver1(T,N,tnum,xnum,hic,hbc,uic,ubc,i0,c,q)    
    h=np.array(h)
    u=np.array(u)
    
    print(u.shape)
    #plt.plot(u)
    
    
    maxu=np.max(u)
    minu=np.min(u)
    
    print(maxu,minu)
    
    n,m=u.shape
    
    
    u_norm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append((u[i,j]-minu)/(maxu-minu))
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
    u_K=np.array(Koopman(t,u_norm,steps))
    print(u_K.shape)
    
    
    u_vnorm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append(u_K[i,j]*(maxu-minu)+minu)
        u_vnorm.append(tem)
    
    u_vnorm=np.array(u_vnorm)
    
    plt.plot(u_K)
    print(np.max(np.abs(u_K-u)))
    #PDEE(t,y,steps)
    np.savetxt('u.txt',u)
    np.savetxt('u_K.txt',u_K)
    