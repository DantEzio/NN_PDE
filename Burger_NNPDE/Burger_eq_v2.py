# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:32:40 2020

@author: Administrator
"""

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
import SV_eq_v2 as SV

class Burger_eq:
    def __init__(self,T,X,rate,Q,A,Sm):
        
        self.Q=Q
        self.A=A
        self.Sm=Sm
        self.u=self.Q/self.A
        
        self.T=T
        self.X=X
        self.xnum,self.tnum=self.Q.shape
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.gama=self.deltt/self.deltx
        
        self.a=2
        self.belt=2.0
        
        self.uic,self.ubc=self.IcBc(rate)
        
        
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
        c=np.ones((self.xnum,self.tnum))
        
        print(self.Q.shape)
        print(c.shape)
        print('xt',self.tnum,self.xnum)
        #r=a*deltt/deltx
        r=deltt/deltx
       
        c[:,0]=self.uic
        c[0,:]=self.ubc

        for n in range(1,self.tnum-1):

            for j in range(1,xnum-2):
                #print(n,j)
                c[j,n+1]=c[j,n]-0.5*r*c[j,n]*(c[j+1,n]-c[j-1,n])\
                    +0.5*r*r*c[j,n]*c[j,n]*(c[j+1,n]-2*c[j,n]+c[j-1,n])\
                    #-0.01*c[j,n]\
                    #+self.Sm[j,n]
                    #+self.belt*0.5*deltt/(deltx*deltx)*(c[n,j+1]-2*c[n,j]+c[n,j-1]) 
                

        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.X/c.shape[1])     
        Y = np.arange(0, self.T, self.T/c.shape[0])
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, c, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        self.c=c
 
    
if __name__=='__main__':
    T=600.6
    tnum=500
    X=20
    xnum=20
    
    sv=SV.SV_eq_v2(T,X,tnum,xnum,0.01,10,0.3)
    sv.sim()
    Sm=np.zeros(sv.Q.shape)
    beq=Burger_eq(T,X,0.2,sv.Q,sv.A,Sm)
    beq.sim()