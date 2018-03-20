# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:34:25 2018

@author: Thomas
"""

# KL UCB project

#import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


#UCB avec 4 bras
K=4

p = np.array([0.05, 0.1, 0.05, 0.05]) #Param des Bernoulli
delta=0.1
Delta = np.array([0.05, 0, 0.05, 0.05]) #Delta avec moy optimale

T= 5000 #Nb max d'iterations

alpha=3

N = np.array([0, 0, 0, 0]) #Nb de fois ou le bras K a été tiré

Reward = np.zeros((K,T+1),dtype=np.int) #Recompense avec chaque bras

Action = np.zeros((K,T+1),dtype=np.int) #Bras choisi a chaque étape

UCB= np.array([10., 10., 10., 10.]) #UCB pour chaque bras

#On commence en explorant les 2 bras au moins une fois
#A t=1

#N[0]=1
#Reward[0,1]= np.random.binomial(1, p[0])
#Action[0,1]=1
#
##A t=2
#N[1]=1
#Reward[1,2]= np.random.binomial(1, p[1])
#Action[1,2]=1

for t in range(1,T+1):
    #Added slight optimization (sum only up to t)
    for i in range(K):
        if (N[i]>0):
            UCB[i] = sum(Reward[i,:t])/N[i]+np.sqrt( (alpha*math.log(t))/(2*N[i]) )
            
    #print("t & UCB :",t,UCB)
    select = np.argmax(UCB)
    #print("t & select",t, select)
    N[select]=N[select]+1
    Action[select,t]=1
    Reward[select,t]= np.random.binomial(1,p[select])

# Now using argmax instead    
#    if UCB[0]>= UCB[1]:
#        N[0]=N[0]+1
#        Reward[0,t]= np.random.binomial(1,p[0])
#    else:
#        N[1]=N[1]+1
#        Reward[1,t]= np.random.binomial(1,p[1])        

print("t & Reward :",t,sum(Reward[0,:]),sum(Reward[1,:]),sum(Reward[0,:])+sum(Reward[1,:]) )


cumReward=np.cumsum(Reward,axis=1) #Reward accumulé en fonction du temps

totalReward=np.sum(cumReward,axis=0)
   
fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
ax1 = fig.add_subplot(1,1,1)
for i in range (K):
##ax1.plot(c_error,marker='.',linestyle='-',label='Online avec gradient à pas constant')
    ax1.plot(cumReward[i,:],linestyle='-',label='Gain avec bras '+str(i))
#ax1.plot(totalReward[1,:],linestyle='-',label='Gain avec bras 2')
ax1.plot(totalReward[:],linestyle='-',label='Gain total UCB')
ax1.legend(loc='best')
plt.show()

totalAction=np.cumsum(Action,axis=1)
#print("t, Nb de tirages sous-optimal, Pseudo-Regret :",t,totalAction[0,t],totalAction[0,t]*delta)
print("t, Pseudo-Regret :",t,np.dot(Delta,totalAction[:,t]))

cumRegret=np.dot(Delta,totalAction[:,:])

fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
ax1 = fig.add_subplot(1,1,1)
##ax1.plot(c_error,marker='.',linestyle='-',label='Online avec gradient à pas constant')
ax1.plot(cumRegret[:],linestyle='-',label='Pseudo-Regret avec UCB')
#ax1.plot(totalReward[1,:],linestyle='-',label='Gain avec bras 2')
#ax1.plot(totalReward[0,:]+totalReward[1,:],linestyle='-',label='Gain total UCB')
ax1.legend(loc='best')
plt.show()


