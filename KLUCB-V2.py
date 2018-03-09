# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:34:25 2018

@author: Thomas
"""

# KL UCB project

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


#D'abord UCB avec 2 bras
K=2 #2 bras qui suivent Bernoulli

p = np.array([0.8, 0.9]) #Param des Bernoulli

T= 5000 #Nb max d'iterations

N = np.array([0, 0]) #Nb de fois ou le bras 1 ou 2 a été tiré

Reward = np.zeros((2,T+1)) #Recompense avec chaque bras

Action = np.zeros((2,T+1)) #Bras choisi a chaque étape

UCB= np.array([0., 0.]) #UCB pour chaque bras

#On commence en explorant les 2 bras au moins une fois
#A t=1

N[0]=1
Reward[0,1]= np.random.binomial(1, p[0])
Action[0,1]=1

#A t=2
N[1]=1
Reward[1,2]= np.random.binomial(1, p[1])
Action[1,2]=1


#A t=3
for t in range(3, T+1):
    UCB[0] = sum(Reward[0,:])/N[0]+np.sqrt( math.log(t)/(2*N[0]) )
    UCB[1] = sum(Reward[1,:])/N[1]+np.sqrt( math.log(t)/(2*N[1]) )
    #print("t & UCB :",t,UCB)
    
    select = np.argmax(UCB)
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


totalReward=np.cumsum(Reward,axis=1) #Reward accumulé en fonction du temps


   
#fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
#ax1 = fig.add_subplot(1,1,1)
##ax1.plot(c_error,marker='.',linestyle='-',label='Online avec gradient à pas constant')
#ax1.plot(totalReward[0,:],linestyle='-',label='Gain avec bras 1')
#ax1.plot(totalReward[1,:],linestyle='-',label='Gain avec bras 2')
#ax1.plot(totalReward[0,:]+totalReward[1,:],linestyle='-',label='Gain total UCB')
#ax1.legend(loc='best')
#plt.show()

totalAction=np.cumsum(Action,axis=1)
fig=plt.figure(figsize=(12,8))
##fig=plt.figure()  
ax1 = fig.add_subplot(1,1,1)
##ax1.plot(c_error,marker='.',linestyle='-',label='Online avec gradient à pas constant')
ax1.plot(totalAction[0,:]*0.1,linestyle='-',label='Pseudo-Regret avec UCB')
#ax1.plot(totalReward[1,:],linestyle='-',label='Gain avec bras 2')
#ax1.plot(totalReward[0,:]+totalReward[1,:],linestyle='-',label='Gain total UCB')
ax1.legend(loc='best')
plt.show()


