#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:37:24 2022

@author: shreyas
"""

'''
Sutton & Barto's example
'''

import numpy as np
import math
import time
from collections import deque
from scipy.special import logsumexp

class Bias():
    def __init__(self, n_iter=10, n_episodes=100000):
        self.N = 1 #State 0
        self.A = 39 #0: Right; 1: Left;
        self.n_iter = n_iter
        self.gamma = 0.99
        self.n_episodes = n_episodes # training episodes

        ##policy can be 'Q' or 'D-Q'
        self.policy = 'Q'
        ##twofold = 0 : original step size; twofold = 1: step size *= 2
        self.twofold = 0
        ##average: =True, use averaged estimator; =False, only use Qa
        self.average = False

        self.rew_arr = np.zeros((self.n_episodes, self.n_iter))
        # initialising Q-table
        self.Qa = np.zeros((self.N, self.A))
        self.Qb = np.zeros((self.N, self.A))


    def initialize_Q(self):

        self.Qa = np.zeros((self.N, self.A))

        self.Qb = np.zeros((self.N, self.A))


    def obtain_max(self):
        curQ = self.Qa

        QvalB = self.Qb

        if (self.policy == 'D-Q') & (self.average == True):
            curQ = (curQ + QvalB) / 2


        return np.max(curQ[0])

    # Choosing action greedily

    def bestAction(self, state, Q):
        return np.argmax(Q[0])

    def obtain_reward(self, state, action):
        if action == 0:  # in State A
            return 0
        else:  # in one of State B
            return np.random.normal(-0.0526, 1)

    def obtain_nextState(self, state, action):  # Return (nextState, Reward, Done?)

        rw = self.obtain_reward(state, action)
        if action == 0:
            return 0, rw, True  # absorbing state
        else:
            return 0, rw, False



    # Choosing action using exploratory policy
    def choose_action(self, Q, state, epsilon):
        
        return np.random.randint(0,39)


    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha, beta, e):

        if self.policy == 'MMQL':  #MMQL 

            weight = 0
            
            eta = (np.log(e+10))**0.5
            
            epk = (e + 10)**2
            
            c = np.max(self.Qa[state_new,:])
            
            lse = (1/epk) * np.log(np.sum(np.exp(epk * self.Qa[state_new,:] - epk * c))) + c # LSE
            
            n = len(self.Qa[state_new, :])     # Number of actions or elements

            # Compute the mellowmax using the exponential trick
            
            mm_value = (1 / eta) * np.log(np.sum(np.exp(eta * self.Qa[state_new, :] - eta * c)) / n) + c # MM
            
            self.Qa[state_old][action] += alpha * (reward + self.gamma * (weight * (lse)+ (1- weight) * (mm_value)) - self.Qa[state_old][action])

        if self.policy == 'WSQL': # WSQL with weight = 0.5
        
            weight = 0.5
            
            eta = (np.log(e+10))**0.5
            
            epk = (e + 10)**2
            
            c = np.max(self.Qa[state_new,:])
            
            lse = (1/epk) * np.log(np.sum(np.exp(epk * self.Qa[state_new,:] - epk * c))) + c
            
            n = len(self.Qa[state_new, :])     # Number of actions or elements

            # Compute the mellowmax using the exponential trick
            mm_value = (1 / eta) * np.log(np.sum(np.exp(eta * self.Qa[state_new, :] - eta * c)) / n) + c
            
            self.Qa[state_old][action] += alpha * (reward + self.gamma * (weight * (lse)+ (1- weight) * (mm_value)) - self.Qa[state_old][action])
        
        
        if self.policy == 'LSEQL': #LSEQL

            weight = 1
            
            eta = (np.log(e+10))**0.5
            
            epk = (e + 10)**2
            
            c = np.max(self.Qa[state_new,:])
            
            lse = (1/epk) * np.log(np.sum(np.exp(epk * self.Qa[state_new,:] - epk * c))) + c
            
            n = len(self.Qa[state_new, :])     # Number of actions or elements

            # Compute the mellowmax using the exponential trick
            mm_value = (1 / eta) * np.log(np.sum(np.exp(eta * self.Qa[state_new, :] - eta * c)) / n) + c
            
            self.Qa[state_old][action] += alpha * (reward + self.gamma * (weight * (lse)+ (1- weight) * (mm_value)) - self.Qa[state_old][action])
        
        if self.policy == 'BZQL': # BZQL
        
            g = (np.log(e+10))**0.5
        
            c = np.max(self.Qa[state_new,:])
            
            
            exp_values = np.exp(g * (self.Qa[state_new, :] - c))  # Exponentiated and stabilized Q-values
            
            softmax_probabilities = exp_values / np.sum(exp_values)  # Softmax probabilities
    
            # Boltzmann softmax expectation
            softmax_expectation = np.sum(softmax_probabilities * self.Qa[state_new, :])
    
            # Q-learning update with softmax expectation
            #self.Qa[state_old][action] += alpha * (reward + self.gamma * softmax_expectation - self.Qa[state_old][action])
            self.Qa[state_old][action] += alpha * (reward + self.gamma * ((softmax_expectation)) - self.Qa[state_old][action])



        if self.policy == 'Q': #Q-learning

            self.Qa[state_old][action] += alpha * (reward + self.gamma * self.Qa[state_new][self.bestAction(state_new,self.Qa)] - self.Qa[state_old][action])


        if self.policy == 'D-Q':
            if np.random.randint(2) < 1: #Double Q-learning

                self.Qa[state_old][action] += alpha * (reward + self.gamma * self.Qb[state_new][self.bestAction(state_new,self.Qa)] - self.Qa[state_old][action])
            else:

                self.Qb[state_old][action] += alpha * (reward + self.gamma * self.Qa[state_new][self.bestAction(state_new,self.Qb)] - self.Qb[state_old][action])



    # Exploration Rate
    def get_epsilon(self, t):
        
        return 0.1

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):

        return 100/ (t + 100)


    def run(self):
        # reward_data = []
        # x_dd = []

        filename = "ProbLeft-"+self.policy
        if self.twofold == 1:
            filename = filename + "-twice"
        if self.average == True:
            filename = filename + "-average"

        print("Running: " + filename)

        for j in range(self.n_iter):

            start_time = time.time()
            self.initialize_Q()

            n_left = 0

            for e in range(self.n_episodes):

                n_left = self.obtain_max()
                self.rew_arr[e, j] = n_left  # save error
                if e % 20 == 0:
                    print("Running: " + filename)
                    print("Iteration, Episode", j, e)
                    print("Running: maxQ(0,a) latest estimate is :", n_left)

                current_state = 0 #Start from State A

                # Get adaptive learning alpha and epsilon decayed over time
                epsilon = self.get_epsilon(e)

                alpha = min(1,(self.twofold + 1) * self.get_alpha(e))
                beta= min(1,self.get_beta(e))
                # beta= self.get_beta(e)
                done = False
                i = 0
                rw = 0
                while not done:
                    # Choose action according to greedy policy and take it
                    curQ = self.Qa
                    if (self.policy == 'D-Q') & (self.average == True):
                        curQ = (self.Qa + self.Qb) / 2


                    current_state = 0

                    action = self.choose_action(curQ, current_state, epsilon)

                    newState, reward, done = self.obtain_nextState(current_state, action)

                    rw = rw + reward

                    # Update Q-Table
                    self.update_q(current_state, action, reward, newState, alpha, beta, e)
                    current_state = newState
                    i += 1

                #calculate performance

            if j % 10 == 0:
                print("Save at iteration: ", j)
                file = open(filename, "wb")
                np.save(file, self.rew_arr)
                file.close()

            total_time = time.time() - start_time
            print("Time taken - %f seconds" % total_time)

        print("Save at iteration: ", j)
        file = open(filename, "wb")
        np.save(file, self.rew_arr)
        file.close()


if __name__ == "__main__":

    #reproduction
    np.random.seed(2020)
    # Make an instance of CartPole class
    solver = Bias()
    solver.run()