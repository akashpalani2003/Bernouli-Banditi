# Task 3
# Using inspiration from code in task1.py and simulator.py write the appropriate functions to create the plot required.

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import *
from task1 import Algorithm
from multiprocessing import Pool
import random


# DEFINE your algorithm class here
class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon,epsilon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = epsilon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t=0
    
    def give_pull(self):
        self.t += 1
        if (self.t<=self.num_arms):
            return self.t-1

        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            result = np.where(self.values == np.max(self.values))[0]
            if(len(result)==1):
                return result[0]
            else:
                max_float = 30000
                j=0
                for i in range(len(result)):
                    a=self.counts[result[i]]
                    if(a<max_float):
                        max_float=a
                        j=i
                    elif (a==max_float):
                        j=random.choice([j,i])

                return result[j]
                #a=np.random.randint(len(result))
                #return result[a]

    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
  
# DEFINE single_sim_task3() HERE
def single_sim_task3(seed=0, ALGO=Eps_Greedy, PROBS=[0.3, 0.5, 0.7,0.4,0.6], HORIZON=30000,e=0.1):
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = BernoulliBandit(probs=shuffled_probs)
  algo_inst = ALGO(num_arms=len(shuffled_probs), horizon=HORIZON,epsilon=e)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()
# DEFINE simulate_task3() HERE
def simulate_task3(algorithm, probs, horizon, num_sims=50,e=0.1):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_task3,
        [(i, algorithm, probs, horizon,e) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets
# DEFINE task3() HERE
def task3(algorithm,probs,horizon):
    e=[0.01*i for i in range(0,101)]
    regrets=[]
    horizon=30000
    num_sims=100
    for i in e :
        print(i)
        regrets.append(simulate_task3(algorithm, probs, horizon, num_sims,i))
    print(regrets)
    min_value = np.min(regrets)      # Minimum value
    min_index = np.argmin(regrets) 
    print("The minimum regret is ", min_value,'at an epsilon value of',min_index*0.01) 
    plt.plot(e, regrets)
    plt.title("Regret vs Epsiolon")
    plt.savefig("task3-{}--{}.png".format(algorithm.__name__,'final1'))
    plt.clf()
# Call task3() to generate the plots
if __name__ == '__main__':
    probs=[0.3,0.4,0.5,0.6,0.7]
    horizon=30000
    task3(Eps_Greedy,probs,horizon)