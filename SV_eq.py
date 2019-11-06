#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:44:44 2019

@author: chong
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SV_eq:
    def __init__(self,T,X,tnum,xnum):
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.Sf=0.02
        self.S0=0.04
        
        self.gama=self.deltt/self.deltx
        
        self.hic,self.hbc,self.Vic,self.Vbc=self.IcBc()
        
    def IcBc(self):
        hic=[]
        hbc=[]
        Vic=[]
        Vbc=[]
        
        for i in range(self.tnum):
            Vbc.append(np.sin(i*np.pi/self.tnum))
            hbc.append(np.cos(i*np.pi/self.tnum))
        
        for j in range(self.xnum):
            if j<=self.xnum/3:
                Vic.append(0.01)
                hic.append(0.1)
            else:
                Vic.append(0.05)
                hic.append(0.05)
        
        #plot
        fig = plt.figure()
        a1=fig.add_subplot(4,1,1)
        a2=fig.add_subplot(4,1,2)
        a3=fig.add_subplot(4,1,3)
        a4=fig.add_subplot(4,1,4)
        
        a1.plot(hic)
        a2.plot(hbc)
        a3.plot(Vic)
        a4.plot(Vbc)
        
        return hic,hbc,Vic,Vbc
    
    
    def P(self,Vt,yt):
        Vtt=[]
        ytt=[]
        for i in range(1,len(Vt)-1):
            
            Va,Vb,Vc=Vt[i-1],Vt[i+1],Vt[i]
            ya,yb,yc=yt[i-1],yt[i+1],yt[i]
            
            if ya<0:
                ya=0
            if yb<0:
                yb=0
            if yc<0:
                yc=0
            
            cc=np.sqrt(10*yc)
            ca=np.sqrt(10*ya)
            cb=np.sqrt(10*yb)
            
            Vr=(Vc+self.gama*(-Vc*ca+cc*Va))/(1+self.gama*(Vc-Va+cc-ca))
            cr=(cc*(1-Vr*self.gama)+ca*Vr*self.gama)/(1+cc*self.gama-ca*self.gama)
            yr=yc-(yc-ya)*(self.gama*(Vr+cr))
            
            #缓流
            Vs=(Vc-self.gama*(Vc*cb-cc*Vb))/(1+self.gama*(Vc-Vb-cc-cb))
            cs=(cc+Vs*self.gama*(cc-cb))/(1+cc*self.gama-cb*self.gama)
            ys=yc-(yc-yb)*(Vs-cs)
            
            #急流
            #Vs=(Vc*(1+self.gama*ca)-Va*cc*self.gama)/(1+self.gama*(Vc-Va+ca-cc))
            #cs=(cc+Vs*self.gama*(ca-cc))/(1+ca*self.gama-cc*self.gama)
            #ys=yc-(yc-ya)*self.gama*(Vs+cs)   
            
            yp=(1/(cr+cs))*(ys*cr+yr*cs+cr*cs*((Vr-Vs)/10))#-self.deltt*(self.Sf-self.Sf))
            
            Vp1=Vr-10*(yp-yr)/(cr)-10*self.deltt*(self.Sf-self.S0)
            Vp2=Vs-10*(yp-ys)/(cs)-10*self.deltt*(self.Sf-self.S0)
            
            Vp=0.5*(Vp1+Vp2)
            
            Vtt.append(Vp)
            ytt.append(yp)
        return Vtt,ytt
    
    def sim(self):
        
        
        v0=self.Vic
        h0=self.hic
        
        V,H=[],[]
        V.append(v0)
        H.append(h0)
        
        #plt.figure()
        for t in range(1,self.tnum):
            vt,ht=self.P(v0,h0)
            vt=[self.Vbc[t]]+vt+[vt[-1]]
            ht=[self.hbc[t]]+ht+[ht[-1]]
            #plt.plot(ht)
            
            V.append(vt)
            H.append(ht)
            
            v0,h0=vt,ht
        
        H=np.mat(H)
        V=np.mat(V)
        
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        '''
        plt.figure()
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        X, Y = np.meshgrid(X, Y)
        plt.contourf(X,Y,np.mat(H),4,cmap=plt.cm.hot)
        plt.show()
        
        plt.figure()
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        X, Y = np.meshgrid(X, Y)
        plt.contourf(X,Y,np.mat(V),4,cmap=plt.cm.hot)
        plt.show()
        
        #plt.figure()
        #plt.plot(H)
        #plt.figure()
        #plt.plot(V)
        '''   
        
class SV_eq_es:
    def __init__(self,T,X,tnum,xnum,V):
        self.T=T
        self.X=X
        self.xnum=xnum
        self.tnum=tnum
        self.deltt=self.T/self.tnum
        self.deltx=self.X/self.xnum
        
        self.V=V
        
        self.hic,self.hbc=self.IcBc()
        
    def IcBc(self):
        hic=[]
        hbc=[]
        
        for i in range(self.tnum):
            hbc.append(np.abs(np.cos(i*np.pi/self.tnum)))
        
        for j in range(self.xnum):
            if j<=self.xnum/3:
                hic.append(0.1)
            else:
                hic.append(0.05)
        
        #plot
        fig = plt.figure()
        a1=fig.add_subplot(4,1,1)
        a2=fig.add_subplot(4,1,2)
        
        a1.plot(hic)
        a2.plot(hbc)
        
        return hic,hbc
    
    def sim(self):
        y0=self.hic
        
        H=[]
        H.append(y0)
        
        #plt.figure()
        for t in range(1,self.tnum):
            ytem=[self.hbc[t]]
            for j in range(1,self.xnum-1):
                a=0.5/self.deltt+0.5*self.V/self.deltx
                b=-0.5/self.deltt+0.5*self.V/self.deltx
                c=0.5/self.deltt-0.5*self.V/self.deltx
                d=0.5/self.deltt+0.5*self.V/self.deltx
                
                y=(b*ytem[j-1]+c*y0[j+1]+d*y0[j])/a
                ytem.append(y)
            ytem.append(y)
            H.append(ytem)
            y0=ytem
            
            
            
        H=np.mat(H)
        #draw V,H
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        #网格化数据
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        
        
        '''
        plt.figure()
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        X, Y = np.meshgrid(X, Y)
        plt.contourf(X,Y,np.mat(H),4,cmap=plt.cm.hot)
        plt.show()
        
        plt.figure()
        X = np.arange(0, self.X, self.deltx)         
        Y = np.arange(0, self.T, self.deltt)
        X, Y = np.meshgrid(X, Y)
        plt.contourf(X,Y,np.mat(V),4,cmap=plt.cm.hot)
        plt.show()
        
        #plt.figure()
        #plt.plot(H)
        #plt.figure()
        #plt.plot(V)
        '''   
            
    
if __name__=='__main__':
    T=2
    tnum=20
    N=50
    xnum=100
    V=20
    sv=SV_eq_es(T,N,tnum,xnum,V)
    sv.sim()
        