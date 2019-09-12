#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:19:40 2019

@author: chong
"""

import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Koopman operator based
'''

def Koopman_test0(t,y,steps):
    #steps=100
    
    n,m=y.shape
    
    x_=tf.placeholder(tf.float32,[None,m])
    y_pre=tf.placeholder(tf.float32,[None,m])
    
    h_size=int(m/3)
        
    #K近似
    W1=tf.Variable(tf.truncated_normal([m,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2=tf.Variable(tf.zeros([h_size]))
    
    W3=tf.Variable(tf.truncated_normal([h_size,m],stddev=0.1))
    b3=tf.Variable(tf.zeros([m]))
    
    h1=tf.nn.tanh(tf.matmul(x_,W1)+b1)
    h2=tf.nn.tanh(tf.matmul(h1,W2)+b2)

    out=tf.nn.tanh(tf.matmul(h2,W3)+b3)#最终输出
    
    loss1=tf.sqrt(tf.reduce_mean(tf.square(out - y_pre)))
    
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #encoding-decoding训练
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array([y[i]])
                #bx.reshape([1,3])
                bypre=np.array([y[i+1]])
                #by.reshape([1,m])
                sess.run(train1,feed_dict={x_:bx,y_pre:bypre})
                
        output1=[]
      
        for i in range(n):
            batch_xs=np.array([y[i]])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(out,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
        
        
        return tv


def Koopman_test01(t,y,steps):
    #steps=100
    
    n,m=y.shape
    
    x_=tf.placeholder(tf.float32,[None,m])
    y_pre=tf.placeholder(tf.float32,[None,m])
    tem_x=tf.reshape(x_,shape=[m,1])
    
    h_size=int(m/3)
        
    #K近似
    W1=tf.Variable(tf.truncated_normal([m,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2=tf.Variable(tf.zeros([h_size]))
    
    W3=tf.Variable(tf.truncated_normal([h_size,m],stddev=0.1))
    b3=tf.Variable(tf.zeros([m]))
    
    h1=tf.nn.tanh(tf.matmul(x_,W1)+b1)
    h2=tf.nn.tanh(tf.matmul(h1,W2)+b2)

    out=tf.nn.tanh(tf.matmul(h2,W3)+b3)#最终输出
    
    loss1t=tf.sqrt(tf.reduce_mean(tf.square(out - y_pre)))
       
    #losspre结合PED能量不等式
    tA=[]
    for i in range(m):
        tem=[]
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(0.0)
                elif j==i+1:
                    tem.append(-0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(0.0)
                elif j==i-1:
                    tem.append(0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(0.5*deltt/deltx)
                elif j==i+1:
                    tem.append(0.0)
                elif j==i+2:
                    tem.append(-0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        tA.append(tem)

    A1=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    
    tA=[]
    for i in range(m):
        tem=[]
        
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i-1:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i+2:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        tA.append(tem)

    A2=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    
    tA=[]
    for i in range(m):
        tem=[]
        
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i-1:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i+2:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        tA.append(tem)

    A3=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    '''
    tA=[]
    for i in range(m-2):
        tem=[]
        for j in range(m):
            if j==i+1:
                tem.append(1.0)
            else:
                tem.append(0.0)
        tA.append(tem)

    C=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    '''
    
    t1=tf.reshape(tf.matmul(A1,tem_x),[1,m])
    t2=tf.reshape(tf.matmul(A2,tem_x),[1,m])
    t3=tf.reshape(tf.multiply(tem_x,tem_x),[m,1])
    
    loss2t=tf.sqrt(tf.reduce_mean(tf.square(tf.matmul(t1,tem_x)+tf.matmul(t2,t3)+tf.matmul(A3,tem_x)+tem_x-y_pre)))
    
    loss1=0.5*loss1t+0.5*loss2t
    
    
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #encoding-decoding训练
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array([y[i]])
                #bx.reshape([1,3])
                bypre=np.array([y[i+1]])
                #by.reshape([1,m])
                sess.run(train1,feed_dict={x_:bx,y_pre:bypre})
                
        output1=[]
      
        for i in range(n):
            batch_xs=np.array([y[i]])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(out,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
        
        
        return tv

def Koopman_test1(t,y,yt,steps):
    #steps=100
    
    n,m=y.shape
    
    x_=tf.placeholder(tf.float32,[None,m])
    y_ed=tf.placeholder(tf.float32,[None,m])
    y_pre=tf.placeholder(tf.float32,[None,m])
    
    h_size=int(m/3)
    
    #encoding-decoding
    W1e=tf.Variable(tf.truncated_normal([m,h_size],stddev=0.1))
    b1e=tf.Variable(tf.zeros([h_size]))
    
    W2e=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2e=tf.Variable(tf.zeros([h_size]))

    W1d=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b1d=tf.Variable(tf.zeros([h_size]))
    
    W2d=tf.Variable(tf.truncated_normal([h_size,m],stddev=0.1))
    b2d=tf.Variable(tf.zeros([m]))
    
    
    #K近似
    W1=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2=tf.Variable(tf.zeros([h_size]))
    
    
    h1=tf.nn.tanh(tf.matmul(x_,W1e)+b1e)
    hout=tf.nn.tanh(tf.matmul(h1,W2e)+b2e)
    h2=tf.nn.tanh(tf.matmul(hout,W1d)+b1d)
    
    hd=tf.nn.tanh(tf.matmul(h2,W2d)+b2d)
    
    
    ht1=tf.nn.tanh(tf.matmul(hout,W1)+b1)
    ht2=tf.nn.tanh(tf.matmul(ht1,W2)+b2)
    
    htout1=tf.nn.tanh(tf.matmul(ht2,W1d)+b1d)
    htout2=tf.nn.tanh(tf.matmul(htout1,W2d)+b2d)#最终输出
    
    loss1=tf.sqrt(tf.reduce_mean(tf.square(hd - y_ed)))+tf.sqrt(tf.reduce_mean(tf.square(htout2 - y_pre)))
    
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #encoding-decoding训练
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array([y[i]])
                #bx.reshape([1,3])
                byed=np.array([y[i]])
                bypre=np.array([y[i+1]])
                #by.reshape([1,m])
                sess.run(train1,feed_dict={x_:bx,y_ed:byed,y_pre:bypre})
                
        output1=[]
      
        for i in range(n):
            batch_xs=np.array([yt[i]])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(htout2,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
        
        
        return tv


def Koopman_test2(t,y,yt,steps,deltt,deltx,belt):
    #steps=100
    
    n,m=y.shape
    print(n,m)
    
    x_=tf.placeholder(tf.float32,[None,m])
    y_ed=tf.placeholder(tf.float32,[None,m])
    y_pre=tf.placeholder(tf.float32,[None,m])
    
    h_size=int(m/3)
    
    tem_x=tf.reshape(x_,shape=[m,1])
    
    #encoding-decoding
    W1e=tf.Variable(tf.truncated_normal([m,h_size],stddev=0.1))
    b1e=tf.Variable(tf.zeros([h_size]))
    
    W2e=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2e=tf.Variable(tf.zeros([h_size]))

    W1d=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b1d=tf.Variable(tf.zeros([h_size]))
    
    W2d=tf.Variable(tf.truncated_normal([h_size,m],stddev=0.1))
    b2d=tf.Variable(tf.zeros([m]))
    
    
    #K近似
    W1=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b1=tf.Variable(tf.zeros([h_size]))
    
    W2=tf.Variable(tf.truncated_normal([h_size,h_size],stddev=0.1))
    b2=tf.Variable(tf.zeros([h_size]))
    
    
    h1=tf.nn.tanh(tf.matmul(x_,W1e)+b1e)
    hout=tf.nn.tanh(tf.matmul(h1,W2e)+b2e)
    h2=tf.nn.tanh(tf.matmul(hout,W1d)+b1d)
    
    hd=tf.nn.tanh(tf.matmul(h2,W2d)+b2d)#encoding输出
    
    
    ht1=tf.nn.tanh(tf.matmul(hout,W1)+b1)
    ht2=tf.nn.tanh(tf.matmul(ht1,W2)+b2)
    
    htout1=tf.nn.tanh(tf.matmul(ht2,W1d)+b1d)
    htout2=tf.nn.tanh(tf.matmul(htout1,W2d)+b2d)#最终输出
    htout2_=tf.reshape(htout2,[m,1])
    
    loss1t=tf.sqrt(tf.reduce_mean(tf.square(hd - y_ed)))+tf.sqrt(tf.reduce_mean(tf.square(htout2 - y_pre)))
       
    #losspre结合PED能量不等式
    tA=[]
    for i in range(m):
        tem=[]
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(0.0)
                elif j==i+1:
                    tem.append(-0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(0.0)
                elif j==i-1:
                    tem.append(0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(0.5*deltt/deltx)
                elif j==i+1:
                    tem.append(0.0)
                elif j==i+2:
                    tem.append(-0.5*deltt/deltx)
                else:
                    tem.append(0.0)
        tA.append(tem)

    A1=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    
    tA=[]
    for i in range(m):
        tem=[]
        
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i-1:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(-deltt*deltt/(deltx*deltx))
                elif j==i+2:
                    tem.append(0.5*deltt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        tA.append(tem)

    A2=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    
    tA=[]
    for i in range(m):
        tem=[]
        
        if i==0:
            for j in range(m):
                if j==i:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        elif i==m-1:
            for j in range(m):
                if j==i:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i-1:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        
        else:
            for j in range(m):
                if j==i:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                elif j==i+1:
                    tem.append(belt*deltt/(deltx*deltx))
                elif j==i+2:
                    tem.append(-0.5*belt*deltt/(deltx*deltx))
                else:
                    tem.append(0.0)
        tA.append(tem)

    A3=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    '''
    tA=[]
    for i in range(m-2):
        tem=[]
        for j in range(m):
            if j==i+1:
                tem.append(1.0)
            else:
                tem.append(0.0)
        tA.append(tem)

    C=tf.cast(tf.constant(np.mat(tA)),tf.float32)
    '''
    
    t1=tf.reshape(tf.matmul(A1,tem_x),[1,m])
    t2=tf.reshape(tf.matmul(A2,tem_x),[1,m])
    t3=tf.reshape(tf.multiply(tem_x,tem_x),[m,1])
    
    loss2t=tf.sqrt(tf.reduce_mean(tf.square(tf.matmul(t1,tem_x)+tf.matmul(t2,t3)+tf.matmul(A3,tem_x)+tem_x-htout2_)))
    
    loss1=0.5*loss1t+0.5*loss2t
    
    train1=tf.train.AdamOptimizer(0.001).minimize(loss1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #encoding-decoding训练
        for j in range(steps):
            for i in range(y.shape[0]-1):
                bx=np.array([y[i]])
                #bx.reshape([1,3])
                byed=np.array([y[i]])
                bypre=np.array([y[i+1]])
                #by.reshape([1,m])
                sess.run(train1,feed_dict={x_:bx,y_ed:byed,y_pre:bypre})
                
        output1=[]
      
        for i in range(n):
            batch_xs=np.array([yt[i]])
            #batch_xs.reshape([1,3])
            #print(batch_xs)
            pre=sess.run(htout2,feed_dict={x_:batch_xs})      
            result=pre#((pre+1)/2)*(outmax[-1]-outmin[-1])+outmin[-1]
            output1.append(result)
        result0=np.array(output1)
        tv=[]
        for i in range(result0.shape[0]):
            #print(abs(result0[i]-y[i]))
            tv.append(result0[i][0])
        
        
        return tv




def o_model_ver1(T,N,tnum,xnum,uic,ubc,a,belt,q,testlag):
  
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
            if testlag==0:
                temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])+0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])
            else:
                temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])+0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])+belt*0.5*deltt/(deltx*deltx)*(u[n][j+1]-2*u[n][j]+u[n][j-1]) 
            ut.append(round(temu,2))
            if j==xnum-2:
                ut.append(round(temu,2))
        u.append(ut)   
    return u



def normalize(u,maxu1,minu1):
    u_norm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append((u[i,j]-minu1)/(maxu1-minu1))
        u_norm.append(tem)
    
    u_norm=np.array(u_norm)
    return u_norm
def vnormalize(u_K,maxu1,minu1):
    u_vnorm=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append(u_K[i,j]*(maxu1-minu1)+minu1)
        u_vnorm.append(tem)
    
    u_vnorm=np.array(u_vnorm)
    return u_vnorm

if __name__=='__main__':
    
    testlag=0
    
    T=20
    tnum=100
    N=50
    xnum=50
    
    hic=[]
    hbc=[]
    uic=[]
    ubc=[]
    
    for i in range(tnum):
        ubc.append(math.sin(i*math.pi/tnum))
        hbc.append(math.cos(i*math.pi/tnum))
    
    for j in range(xnum):
        if j<=xnum/3:
            uic.append(2.0)
        else:
            uic.append(1.0)
    
    
    q=[]
    for i in range(tnum):
        qtem=[]
        for j in range(xnum):
            qtem.append(0)
        q.append(qtem)
    
    q=np.array(q)
    
    i0=0.5
    c=0.5
    a=2
    belt=2.0
    
    u=o_model_ver1(T,N,tnum,xnum,uic,ubc,a,belt,q,testlag)    
    #h=np.array(h).T
    u=np.array(u).T
    print(u.shape)
    #plt.plot(u)
    
    n,m=u.shape
    
    maxu1=np.max(u)
    minu1=np.min(u)
    
    print(maxu1,minu1)
    
    
    u_norm=normalize(u,maxu1,minu1)
    
    fig = plt.figure()
    a1=fig.add_subplot(5,1,1)
    a2=fig.add_subplot(5,1,2)
    a3=fig.add_subplot(5,1,3)
    a4=fig.add_subplot(5,1,4)
    a5=fig.add_subplot(5,1,5)
    
    a1.plot(u)
    
    #plt.plot(u_norm)
    deltx=N/xnum
    deltt=T/tnum
    
    t=[]
    for i in range(tnum):
        t.append(i*deltt)
    
    t=np.array(t)
    
    steps=100
    u_K0=np.array(Koopman_test0(t,u_norm,steps))
    u_K1=np.array(Koopman_test01(t,u_norm,steps))
    u_K2=np.array(Koopman_test1(t,u_norm,u_norm,steps))
    u_K3=np.array(Koopman_test2(t,u_norm,u_norm,steps,deltt,deltx,belt))
    
    uv0=vnormalize(u_K0,maxu1,minu1)
    uv1=vnormalize(u_K1,maxu1,minu1)
    uv2=vnormalize(u_K2,maxu1,minu1)
    uv3=vnormalize(u_K2,maxu1,minu1)

    a2.plot(uv0)
    a3.plot(uv1)
    a4.plot(uv2)
    a5.plot(uv3)
    #PDEE(t,y,steps)
    