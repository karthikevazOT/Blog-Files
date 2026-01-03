#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 07:25:36 2025

@author: karthikelamvazhuthi
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import imageio.v2 as imageio


N = 1 # Number of samples
x0 = np.random.rand(2,N)*2-1
teval = np.arange(0,100,0.1)
Nt = len(teval)
xt = np.zeros((2,len(teval),N))


kde = KernelDensity(
      kernel = "gaussian",
      bandwidth = 0.3
      )




def van_der(t,y):
    dy = np.zeros(2)
    dy[0] = y[1]
    dy[1] = -1*(y[0]**2-1)*dy[0]-y[0]
    return dy


    
for i  in range(N):
  sol = solve_ivp(van_der,[0,100],x0[:,i],t_eval = teval)    
  xt[:,:,i] = sol.y
  

#%%
frames = []
fig, ax = plt.subplots(figsize=(5, 3), dpi=120)

for it in range(0,120):
    ax.clear()
    ax.plot(xt[0, it, :], xt[1, it, :], 's')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())     # (h, w, 4)
    frames.append(buf[..., :3].copy())             # (h, w, 3)

plt.close(fig)
imageio.mimsave("single.gif", frames, fps=12,loop = 0)

#%%
frames = []
fig, ax = plt.subplots(figsize=(5, 3), dpi=120)

for it in range(0,100):     
    
    ax.clear()
    samples  = xt[:,it,:].T
    kde.fit(samples)
    
    xmin, xmax = -3, 3
    ymin, ymax = -3, 3
    
    
    nx = 100
    ny = 100
    
    xx, yy  = np.meshgrid(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny))
    
    grid = np.c_[xx.ravel(),yy.ravel()]
    score = kde.score_samples(grid)    
    dens = np.exp(score)
    
    
    #plt.scatter(X[:,0],X[:,1], s = 8,alpha = 0.4)
    ax.contourf(xx,yy,dens.reshape(xx.shape),levels  = 20, vmin = 0.0, vmax = 0.1)        
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf[..., :3].copy())             # (h, w, 3)
    
   
plt.close(fig)
imageio.mimsave("kde.gif", frames, fps=15,loop = 0)    
    
 #%%
   

      
  
