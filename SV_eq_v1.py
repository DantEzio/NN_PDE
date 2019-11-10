#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:44:44 2019

@author: chong
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
        
class SV_eq_es:
    def __init__(self,T,X,tnum,xnum,n,R):
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.n=n
        self.R=R
        
        self.hic,self.hbc,self.Qic,self.Qbc=self.IcBc()
        
    def IcBc(self):
        Qic=[]
        Qbc=[]
        hic=[]
        hbc=[]
        for i in range(self.tnum):
            if i < self.tnum/3:
                Qbc.append(100.0)
            else:
                Qbc.append(10.0)
            
            hbc.append(0.01)
        
        for j in range(self.xnum*2+1):
            if j<=(self.xnum*2+1)/3:
                Qic.append(20.0)
            else:
                Qic.append(1.1)
            hic.append(10.0)
        return np.array(hic),np.array(hbc),np.array(Qic),np.array(Qbc)
    
    def sim(self):
        
        A=np.zeros((self.xnum*2+1,self.tnum))
        Q=np.zeros((self.xnum*2+1,self.tnum))
        u=np.zeros((self.xnum*2+1,self.tnum))
        Z=np.zeros((self.xnum*2+1,self.tnum))
        b=np.ones((self.xnum*2+1,1))
        
        Q[:,0]=self.Qic
        Q[:,1]=self.Qic
        Q[0,:]=self.Qbc
        Z[:,0]=self.hic
        Z[:,1]=self.hic
        Z[0,:]=self.hbc
        
        
        A=np.ones((self.xnum*2+1,self.tnum))
        Q=np.ones((self.xnum*2+1,self.tnum))
        u=np.ones((self.xnum*2+1,self.tnum))
        Z=np.ones((self.xnum*2+1,self.tnum))
        b=np.ones((self.xnum*2+1,1))
        
        A[:,0]=Z[:,0]*b.T
        A[:,1]=Z[:,1]*b.T
        A[0,:]=Z[0,:]*b[0]
        
        u[:,0]=Q[:,0]/A[:,0]
        u[:,1]=Q[:,1]/A[:,1]
        u[0,:]=Q[0,:]/A[0,:]
        
        for t in range(2,self.tnum-1):
            
            for i in range(2,self.xnum-2):
                
                self.deltt=0.1*self.deltx/(np.abs(u[i,t])+np.sqrt(10*A[i,t]/b[i]))
                
                
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
                    print('wrong',t,i)
                    #A[i,t+1]=1.01#np.abs(A[i,t+1])
                ###############################################################
                II=0.0
                if Q[i,t]>0 or Q[i,t]==0:
                    II=((Q[i,t]/A[i,t])**2-0.5*(Q[i-2,t]/A[i-2,t])**2)/self.deltx
                else:
                    II=((Q[i+2,t]/A[i+2,t])**2-0.5*(Q[i,t]/A[i,t])**2)/self.deltx
                #print('ii:',II)
                
                '''
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
                '''
                
                III=(Z[i+1,t]-Z[i-1,t])/(4*self.deltx)
                
                #print('iii:',III)
                ###############################################################
                a=Q[i,t]-self.deltt*II-10*A[i,t]*III*self.deltt
                am=1+((10*self.n**2*np.abs(Q[i,t])*self.deltt)/(np.power(self.R,4/3)*A[i,t]))
                Q[i,t+1]=a/am
                
                Z[i,t+1]=A[i,t+1]/b[i]
                u[i,t+1]=Q[i,t+1]/A[i,t+1]
                
            A[A.shape[1]-1,t+1]=A[A.shape[1]-3,t+1]
            Q[Q.shape[1]-1,t+1]=Q[Q.shape[1]-3,t+1]
            Z[Z.shape[1]-1,t+1]=Z[Z.shape[1]-3,t+1]
            u[u.shape[1]-1,t+1]=u[u.shape[1]-3,t+1]
            
            A[A.shape[1]-2,t+1]=A[A.shape[1]-3,t+1]
            Q[Q.shape[1]-2,t+1]=Q[Q.shape[1]-3,t+1]
            Z[Z.shape[1]-2,t+1]=Z[Z.shape[1]-3,t+1]
            u[u.shape[1]-2,t+1]=u[u.shape[1]-3,t+1]
        
        
        #print(Z.shape,Q.shape)
        
        
        #draw Z,Q
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/Z.shape[0])         
        Y = np.arange(0, self.T, self.T/Z.shape[1])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Z.T, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
        
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/Q.shape[0])         
        Y = np.arange(0, self.T, self.T/Q.shape[1])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Q.T, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
        #plt.figure()
        #plt.plot(H)
        #plt.figure()
        #plt.plot(V)
           
            
    
if __name__=='__main__':
    T=40.6
    tnum=90
    N=50
    xnum=50
    n=0.02
    R=0.05
    sv=SV_eq_es(T,N,tnum,xnum,n,R)
    sv.sim()
        