# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:34:25 2018

@author: Thomas
"""

# Algo UCB pour scenario 2

#import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Params du scenario 2
p = np.array([0.05, 0.02, 0.01, 0.05, 0.02, 0.01, 0.05, 0.02, 0.01, 0.1]) #Param des Bernoulli
K= p.shape[0] #Nb de bras
p_star = np.max(p)
Delta= ( np.ones(K)*p_star ) - p

T= 50000 #Nb max d'iterations


#UCB params initialization
alpha=3
N3 = np.zeros(K) #Nb de fois ou le bras K a été tiré

Reward_UCB3 = np.zeros((K,T),dtype=np.int) #Recompense avec chaque bras
Action_UCB3 = np.zeros((K,T),dtype=np.int) #Bras choisi a chaque étape

UCB3 = np.zeros(K) #Gain UCB pour chaque bras

#On commence en explorant les K bras au moins une fois
for t in range(K):
    N3[t]=1
    Reward_UCB3[t,t]= np.random.binomial(1, p[t])
    Action_UCB3[t,t]=1

for t in range(K,T):
    #Added slight optimization (sum only up to t)
    for i in range(K):
        UCB3[i] = np.sum(Reward_UCB3[i,:t])/N3[i]+np.sqrt( (alpha*math.log(t))/(2*N3[i]) )
            
    select = np.argmax(UCB3)
    #print("t & select",t, select)
    N3[select]=N3[select]+1
    Action_UCB3[select,t]=1
    Reward_UCB3[select,t]= np.random.binomial(1,p[select])

#print("t & Reward :",t,sum(Reward[0,:]),sum(Reward[1,:]),sum(Reward[0,:])+sum(Reward[1,:]) )
cumReward=np.cumsum(Reward_UCB3,axis=1) #Reward accumulé pour chaque bras en fonction du temps
totalReward=np.sum(cumReward,axis=0) #Reward sur tout les bras accumulé en fonction du temps

print("t & Reward (total) :",t, totalReward[t-1] )
   
fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
ax1 = fig.add_subplot(1,1,1)
for i in range (K):
##ax1.plot(c_error,marker='.',linestyle='-',label='Online avec gradient à pas constant')
    ax1.plot(cumReward[i,:],linestyle='-',label='Gain avec bras '+str(i))
#ax1.plot(totalReward[1,:],linestyle='-',label='Gain avec bras 2')
ax1.plot(totalReward[:],linestyle='-',label='Gain total')
ax1.legend(loc='best')
plt.show()

totalAction=np.cumsum(Action_UCB3,axis=1) #Nb d'actions accumulé sur chaque bras
cumRegret=np.dot(Delta,totalAction[:,:]) #Regret accumulé

print("t & Pseudo-Regret :",t,cumRegret[T-1])

fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
ax1 = fig.add_subplot(1,1,1)
ax1.plot(cumRegret[:],linestyle='-',label='Pseudo-Regret avec UCB avec alpha=3')
ax1.legend(loc='best')
ax1.grid()
#ax1.xscale('log')
plt.show()


