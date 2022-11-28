# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import random
import time

import numpy as np

from matplotlib import pyplot as plt

import gym
from gym import spaces



###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters for the environment configuration
size = 3
timeOut = 10

# Parameters associated with the environment stochasticity (risk)
probaWind = 0.5



###############################################################################
############################ Class RiskyTransitions ###########################
###############################################################################

class RiskyTransitions(gym.Env):
    """
    GOAL: Implementing a simple RL environment consisting of a 2D grid world
          where the agent has to reach one of the two objective locations,
          with a stochastinc wind randomly pushing the agent to the left.
    
    VARIABLES: - observation_space: RL environment observation space.
               - action_space: RL environment action space.
               - playerPosition: Position of the player (x, y).
               - targetPosition0: Position of the first target (x, y).
               - targetPosition1: Position of the second target (x, y).
               - timeElapsed: Time elapsed (number of time steps).
               - state: RL state or observation.
               - reward: RL reward signal.
               - done: RL termination signal (end of episode).
               - info: Additional RL information.
                                
    METHODS: - __init__: Initialization of the RL environment.
             - reset: Resetting of the RL environment.
             - step: Update the RL environment according to the agent's action.
             - render: Render graphically the current state of the RL environment.
    """

    def __init__(self, size=size):
        """
        GOAL: Perform the initialization of the RL environment.
        
        INPUTS: - size: Size of the square grid world.
        
        OUTPUTS: /
        """

        super(RiskyTransitions, self).__init__()

        # Initialization of the random function with a variable random seed
        random.seed(time.time())

        # Definition of the observation/state and action spaces
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.size = size

        # Initialization of the player position
        self.playerPosition = [int(self.size/2), int(self.size/2)-1]

        # Initialization of the two objective locations
        self.targetPosition0 = [0, self.size-1]
        self.targetPosition1 = [self.size-1, 0]

        # Initialization of the time elapsed
        self.timeElapsed = 0

        # Initialization of the RL variables
        self.state = np.array([self.playerPosition[0], self.playerPosition[1]])
        self.reward = 0.
        self.done = 0
        self.info = {}


    def reset(self):
        """
        GOAL: Perform a reset of the RL environment.
        
        INPUTS: /
        
        OUTPUTS: - state: RL state or observation.
        """

        # Reset of the player position and time elapsed
        self.playerPosition = [int(self.size/2), int(self.size/2)-1]
        self.timeElapsed = 0

        # Reset of the RL variables
        self.state = np.array([self.playerPosition[0], self.playerPosition[1]])
        self.reward = 0.
        self.done = 0
        self.info = {}

        return self.state


    def step(self, action):
        """
        GOAL: Update the RL environment according to the agent's action.
        
        INPUTS: - action: RL action outputted by the agent.
        
        OUTPUTS: - state: RL state or observation.
                 - reward: RL reward signal.
                 - done: RL termination signal.
                 - info: Additional RL information.
        """

        # Movement of the agent according to the selected action
        # Go right
        if action == 0:
            self.playerPosition[0] = min(self.playerPosition[0]+1, self.size-1)
        # Go down
        elif action == 1:
            self.playerPosition[1] = max(self.playerPosition[1]-1, 0)
        # Go left
        elif action == 2:
            self.playerPosition[0] = max(self.playerPosition[0]-1, 0)
        # Go up
        elif action == 3:
            self.playerPosition[1] = min(self.playerPosition[1]+1, self.size-1)
        # Invalid action
        else:
            print("Error: invalid action...")

        # Stochastic transitions because of a random wind
        if random.random() < probaWind:
            self.playerPosition[0] = max(self.playerPosition[0]-1, 0)
            
        # Assignation of the RL reward
        if self.playerPosition == self.targetPosition0 or self.playerPosition == self.targetPosition1:
            self.reward = np.random.normal(loc=1.0, scale=0.1)
            self.done = 1
            if self.playerPosition == self.targetPosition0:
                self.info = {"riskSensitivePolicy" : 1}
            else:
                self.info = {"riskSensitivePolicy" : -1}
        else:
            self.reward = np.random.normal(loc=-0.3, scale=0.1)

        # Check if the time elapsed reaches the time limit
        self.timeElapsed += 1
        if self.timeElapsed >= timeOut:
            self.done = 1
            self.info = {"riskSensitivePolicy" : 0}

        # Update of the RL state
        self.state = np.array([self.playerPosition[0], self.playerPosition[1]])

        # Return of the RL variables
        return self.state, self.reward, self.done, self.info

    
    def render(self, mode='human'):
        """
        GOAL: Render graphically the current state of the RL environment.
        
        INPUTS: /
        
        OUTPUTS: /
        """

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, self.size+1, 1))
        ax.set_yticks(np.arange(0, self.size+1, 1))
        ax.set(xlim=(0, self.size), ylim=(0, self.size))
        plt.scatter(self.playerPosition[0]+0.5, self.playerPosition[1]+0.5, s=100, color='blue')
        plt.scatter(self.targetPosition0[0]+0.5, self.targetPosition0[1]+0.5, s=100, color='green')
        plt.scatter(self.targetPosition1[0]+0.5, self.targetPosition1[1]+0.5, s=100, color='red')
        plt.grid()
        text = ''.join(['Time elapsed: ', str(self.timeElapsed)])
        plt.text(0, self.size+0.2, text, fontsize=12)
        plt.show()
        #plt.savefig("RiskyTransitionsEnvironment.pdf", format="pdf")


    def setState(self, state):
        """
        GOAL: Reset the RL environment and set a specific initial state.
        
        INPUTS: - state: Information about the state to set.
        
        OUTPUTS: - state: RL state of the environment.
        """

        # Reset of the environment
        self.reset()

        # Set the initial state as specified
        self.timeElapsed = 0
        self.playerPosition = [state[0], state[1]]
        self.state = np.array([self.playerPosition[0], self.playerPosition[1]])

        return self.state
        
