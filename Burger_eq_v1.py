# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:13:44 2020

@author: Administrator
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:51:11 2019

@author: chong
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Burger_eq:
    def __init__(self,T,X,tnum,xnum,l,rate):
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.gama=self.deltt/self.deltx
        
        self.a=2
        self.belt=2.0
        
        self.uic,self.ubc=self.IcBc(rate)
        
        self.testlag=l
        
    def IcBc(self,rate):
        uic=[]
        ubc=[]
        
        for i in range(self.tnum):
            #ubc.append(np.sin(i*np.pi/self.tnum))
            
            if i >= self.tnum*rate and i<=self.tnum*(0.2+rate):
                ubc.append(1.0)
            else:
                ubc.append(0.0)
            
            
        
        for j in range(self.xnum):
            if j<=self.xnum/3:
                uic.append(1.0)
            else:
                uic.append(0.0)
        
        #plot
        #fig = plt.figure()
        #a1=fig.add_subplot(2,1,1)
        #a2=fig.add_subplot(2,1,2)
        #a1.plot(uic)
        #a2.plot(ubc)
        
        return uic,ubc
    
    def sim(self):
  
        deltx=self.X/self.xnum
        deltt=self.T/self.tnum
        
        #r=a*deltt/deltx
        r=deltt/deltx
        u=[]
        
        u.append(self.uic)
        
        
        for n in range(self.tnum-1):
            ut=[]
            ut.append(self.ubc[n])
            for j in range(1,xnum-1):
                if self.testlag==0:
                    temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])\
                    +0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])
                else:
                    temu=u[n][j]-0.5*r*u[n][j]*(u[n][j+1]-u[n][j-1])\
                    +0.5*r*r*u[n][j]*u[n][j]*(u[n][j+1]-2*u[n][j]+u[n][j-1])\
                    +self.belt*0.5*deltt/(deltx*deltx)*(u[n][j+1]-2*u[n][j]+u[n][j-1]) 
                ut.append(round(temu,2))
                if j==self.xnum-2:
                    ut.append(round(temu,2))
            u.append(ut)   
                
        u=np.mat(u)
        print(u.shape)
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        self.u=u
 
    
if __name__=='__main__':
    T=20
    tnum=100
    X=50
    xnum=50
    l=1
    beq=Burger_eq(T,X,tnum,xnum,l,0.2)
    beq.sim()