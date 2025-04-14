"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
def log(x):
    return math.log(x)

def KL(p,q):
    x=1e-7
    if(p==0):
        prob_p=x
    elif(p==1):
        prob_p=1-x
    else:
        prob_p=p
    if(q==0):
        prob_q=x
    elif(q==1):
        prob_q=1-x
    else:
        prob_q=q 
    a=prob_p*log(prob_p/prob_q)
    b=(1-prob_p)*log((1-prob_p)/(1-prob_q))
    return a+b



def q_search1(p_hat,z,x,e=1e-6):
    l=p_hat
    h=z
    while (h-l>e):
        q=(l+h)/2
        k=KL(p_hat,q)
        error=x-k
        if (abs(error)<e):
            return q
        elif(error>0):
            l=q
        else:
            h=q
    return l/2+h/2

def q_search_newton(p_hat,x,e=1e-4):
    q=p_hat/2+1/2
    k=KL(p_hat,q)
    error=x-k
    i=0
    while(abs(error)>e and i<50):
        i=i+1
        if q <= 0 or q >= 1:
            break
        d=p_hat/q-(1-p_hat)/(1-q)
        if (abs(d)<e):
            break
        q=q-error/(d)
        q = max(min(q, 1 - 1e-7), 1e-7)
        error=x-KL(p_hat,q)
    return q
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts=np.zeros(num_arms)
        self.values=np.zeros(num_arms)
        self.ucb=np.zeros(num_arms)
        self.t=0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.t += 1
        if (self.t<=self.num_arms):
            return self.t-1
        sqrt_ln_t=math.sqrt(log(self.t))
        
        for i in range(self.num_arms):
            sqrt_u=math.sqrt(self.counts[i])
            self.ucb[i]=self.values[i]+ sqrt_ln_t/sqrt_u
        result = np.where(self.ucb == np.max(self.ucb))[0]
        if(len(result)==1):
            return result[0]
        else:

            a=np.random.randint(len(result))
            return result[a]
        
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = (value*(n-1)+reward)/n
        self.values[arm_index]=new_value

        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts=np.zeros(num_arms)
        self.values=np.zeros(num_arms)
        self.kl_ucb=np.zeros(num_arms)
        self.c=3
        self.t=0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.t +=1
        if (self.t<=self.num_arms):
            return self.t-1
        x=(log(self.t)+self.c*log(log(self.t)))
        for i in range(self.num_arms):
            y=x/self.counts[i]
            p=self.values[i]
            self.kl_ucb[i]=q_search_newton(p,y)
        result = np.where(self.kl_ucb == np.max(self.kl_ucb))[0]
        if(len(result)==1):
            return result[0]
        else:

            a=np.random.randint(len(result))
            return result[a]  
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = (value*(n-1)+reward)/n
        self.values[arm_index]=new_value        
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.s_counts=np.zeros(num_arms)
        self.f_counts=np.zeros(num_arms)
        self.thom=np.zeros(num_arms)
        self.t=0        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.t += 1
        for i in range(self.num_arms):
            self.thom[i]=np.random.beta(self.s_counts[i]+1,self.f_counts[i]+1)
        result = np.where(self.thom == np.max(self.thom))[0]
        if(len(result)==1):
            return result[0]
        else:

            a=np.random.randint(len(result))
            return result[a]         
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward==1:
            self.s_counts[arm_index] += 1
        elif reward==0:
            self.f_counts[arm_index] +=1
        # END EDITING HERE

