#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    
    
    x_=tf.placeholder(tf.float32,[None,1])
    y_=tf.placeholder(tf.float32,[None,1])
    
    h_size=10
    
    W1=tf.Variable(tf.truncated_normal([1,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W3=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b3=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,1],stddev=0.1))
    b2=tf.Variable(tf.zeros([1]))
    
    h1=tf.nn.tanh(tf.matmul(x_,W1)+b1)
    h2=tf.nn.tanh(tf.matmul(h1,W3)+b3)
    
    out_=tf.nn.tanh(tf.matmul(h2,W2)+b2)
    
    loss1=tf.sqrt(tf.reduce_mean(tf.square(out_ - y_)))
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array(y[i])
                #bx.reshape([1,3])
                by=np.array(y[i+1])
                by.reshape([1,1])
                sess.run(train1,feed_dict={x_:bx,y_:by})
                
        output1=[]
        for i in range(y.shape[0]):
            batch_xs=np.array(y[i])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(out_,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        p=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
            p.append(y[i][0])
        
        plt.plot(t,tv)
        plt.plot(t,p)


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
        
        plt.plot(t,tv)
        plt.plot(t,p)   
    



if __name__=='__main__':
    t=np.array([math.pi*2*(i/100) for i in range(100)])
    y=[]
    y_bar=[]
    for i in range(100):
        y.append([[-math.cos(math.pi*2*(i/100))]])#,-math.cos(i+1),-math.cos(i+2)])
        y_bar.append([[-math.sin(math.pi*2*(i/100))]])
    y=np.array(y)
    y_bar=np.array(y_bar)
    steps=10
    Koopman(t,y,steps)
    PDEE(t,y,steps)
    