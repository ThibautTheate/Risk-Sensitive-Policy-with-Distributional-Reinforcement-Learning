# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import math
import random
import copy
import datetime
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from replayMemory import ReplayMemory

from Models.FeedforwardDNN import FeedForwardDNN



###############################################################################
#################################### Class DQN ################################
###############################################################################

class DQN:
    """
    GOAL: Implementing the DQN Deep Reinforcement Learning algorithm.
    
    VARIABLES: - device: Hardware specification (CPU or GPU).
               - gamma: Discount factor of the RL algorithm.
               - learningRate: Learning rate of the DL optimizer (ADAM).
               - epsilon: Epsilon value for the DL optimizer (ADAM).
               - targetNetworkUpdate: Update frequency of the target network.
               - learningUpdatePeriod: Frequency of the learning procedure.
               - batchSize: Size of the batch to sample from the replay memory.
               - capacity: Capacity of the replay memory.
               - replayMemory: Experience Replay memory.
               - rewardClipping: Clipping of the RL rewards.
               - gradientClipping: Clipping of the training loss.
               - optimizer: DL optimizer (ADAM).
               - epsilonStart: Initial value of epsilon (Epsilon-Greedy).
               - epsilonEnd: Final value of epsilon (Epsilon-Greedy).
               - epsilonDecay: Exponential decay of epsilon (Epsilon-Greedy).
               - epsilonTest: Test value of epsilon (Epsilon-Greedy).
               - epsilonValue: Current value of epsilon (Epsilon-Greedy).
               - policyNetwork: Deep Neural Network representing the info used by the RL policy.
               - targetNetwork: Deep Neural Network representing the target network.
               - iterations: Counter of the number of iterations.
                                
    METHODS: - __init__: Initialization of the DQN algorithm.
             - readParameters: Read the JSON file to load the parameters.
             - initReporting: Initialize the reporting tools.
             - processState: Process the RL state returned by the environment.
             - processReward: Process the RL reward returned by the environment.
             - updateTargetNetwork: Update the target network (parameters transfer).
             - chooseAction: Choose a valid action based on the current state
                             observed, according to the RL policy learned.
             - chooseActionEpsilonGreedy: Choose a valid action based on the
                                          current state observed, according to
                                          the RL policy learned, following the 
                                          Epsilon Greedy exploration mechanism.
             - fillReplayMemory: Fill the replay memory with random experiences before
                                 the training procedure begins.
             - learning: Execute the DQN learning procedure.
             - training: Train the DQN agent by interacting with the environment.
             - testing: Test the DQN agent learned policy on the RL environment.
             - saveModel: Save the RL policy learned.
             - loadModel: Load a RL policy.
             - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(self, observationSpace, actionSpace, environment,
                 parametersFileName='', reporting=True):
        """
        GOAL: Initializing the RL agent based on the DQN Deep Reinforcement Learning
              algorithm, by setting up the algorithm parameters as well as 
              the Deep Neural Networks.
        
        INPUTS: - observationSpace: RL observation space.
                - actionSpace: RL action space.
                - environment: Name of the RL environment.
                - parametersFileName: Name of the JSON parameters file.
                - reporting: Enable the reporting of the results.
        
        OUTPUTS: /
        """

        # Initialize the random function with a fixed random seed
        random.seed(0)

        # Setting of the parameters
        if parametersFileName == '':
            parametersFileName = ''.join(['Parameters/parameters_DQN_', str(environment), '.json'])
        parameters = self.readParameters(parametersFileName)

        # Set the device for DNN computations (CPU or GPU)
        self.device = torch.device('cuda:'+str(parameters['GPUNumber']) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the DQN algorithm
        self.gamma = parameters['gamma']
        self.learningRate = parameters['learningRate']
        self.epsilon = parameters['epsilon']
        self.targetUpdatePeriod = parameters['targetUpdatePeriod']
        self.learningUpdatePeriod = parameters['learningUpdatePeriod']
        self.rewardClipping = parameters['rewardClipping']
        self.gradientClipping = parameters['gradientClipping']

        # Set the Experience Replay mechanism
        self.batchSize = parameters['batchSize']
        self.capacity = parameters['capacity']
        self.replayMemory = ReplayMemory(self.capacity)

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.policyNetwork = FeedForwardDNN(observationSpace, actionSpace, parameters['structureDNN']).to(self.device)
        self.targetNetwork = FeedForwardDNN(observationSpace, actionSpace, parameters['structureDNN']).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonStart = parameters['epsilonStart']
        self.epsilonEnd = parameters['epsilonEnd']
        self.epsilonDecay = parameters['epsilonDecay']
        self.epsilonTest = parameters['epsilonTest']
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)
        
        # Initialization of the counter for the number of steps
        self.steps = 0

        # Initialization of the experiment folder and tensorboard writer
        if reporting:
            self.initReporting(parameters, 'DQN')


    def readParameters(self, fileName):
        """
        GOAL: Read the appropriate JSON file to load the parameters.
        
        INPUTS: - fileName: Name of the JSON file to read.
        
        OUTPUTS: - parametersDict: Dictionary containing the parameters.
        """

        # Reading of the parameters file, and conversion to Python disctionary
        with open(fileName) as parametersFile:
            parametersDict = json.load(parametersFile)
        return parametersDict

    
    def initReporting(self, parameters, algorithm='DQN'):
        """
        GOAL: Initialize both the experiment folder and the tensorboard
              writer for reporting (and storing) the results.
        
        INPUTS: - parameters: Parameters to ne stored in the experiment folder.
                - algorithm: Name of the RL algorithm.
        
        OUTPUTS: /
        """

        while True:
            try:
                time = datetime.datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
                self.experimentFolder = ''.join(['Experiments/', algorithm, '_', time, '/'])
                os.mkdir(self.experimentFolder)
                with open(''.join([self.experimentFolder , 'Parameters.json']), "w") as f:  
                    json.dump(parameters, f, indent=4)
                self.writer = SummaryWriter(''.join(['Tensorboard/', algorithm, '_', time]))
                break
            except:
                pass
    
    
    def processState(self, state):
        """
        GOAL: Potentially process the RL state returned by the environment.
        
        INPUTS: - state: RL state returned by the environment.
        
        OUTPUTS: - state: RL state processed.
        """

        return state

    
    def processReward(self, reward):
        """
        GOAL: Potentially process the RL reward returned by the environment.
        
        INPUTS: - reward: RL reward returned by the environment.
        
        OUTPUTS: - reward: RL reward processed.
        """

        return np.clip(reward, -self.rewardClipping, self.rewardClipping)
 

    def updateTargetNetwork(self):
        """
        GOAL: Taking into account the update frequency (parameter), update the
              target Deep Neural Network by copying the policy Deep Neural Network
              parameters (weights, bias, etc.).
        
        INPUTS: /
        
        OUTPUTS: /
        """

        # Check if an update is required (update frequency)
        if(self.steps % self.targetUpdatePeriod == 0):
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        

    def chooseAction(self, state, plot=False):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
                - plot: Enable the plotting of information about the decision.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            QValues = self.policyNetwork(state).squeeze(0)
            _, action = QValues.max(0)

            # If required, plot the expected return Q associated with each action
            if plot:
                colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown']
                fig = plt.figure()
                ax = fig.add_subplot()
                QValues = QValues.cpu().numpy()
                for a in range(self.actionSpace):
                    ax.axvline(x=QValues[a], linewidth=5, label=''.join(['Action ', str(a), ' expected return Q']), color=colors[a])
                ax.set_xlabel('Expected return Q')
                ax.set_ylabel('')
                ax.legend()
                plt.show()

            return action.item()

    
    def chooseActionEpsilonGreedy(self, state, epsilon):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed, following the 
              Epsilon Greedy exploration mechanism.
        
        INPUTS: - state: RL state returned by the environment.
                - epsilon: Epsilon value from Epsilon Greedy technique.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # EXPLOITATION -> RL policy
        if(random.random() > epsilon):
            action = self.chooseAction(state)
        # EXPLORATION -> Random
        else:
            action = random.randrange(self.actionSpace)

        return action


    def fillReplayMemory(self, trainingEnv):
        """
        GOAL: Fill the experiences replay memory with random experiences before the
              the training procedure begins.
        
        INPUTS: - trainingEnv: Training RL environment.
                
        OUTPUTS: /
        """

        # Fill the replay memory with random RL experiences
        while self.replayMemory.__len__() < self.capacity:

            # Set the initial RL variables
            state = self.processState(trainingEnv.reset())
            done = 0

            # Interact with the training environment until termination
            while done == 0:

                # Choose an action according to the RL policy and the current RL state
                action = random.randrange(self.actionSpace)
                
                # Interact with the environment with the chosen action
                nextState, reward, done, info = trainingEnv.step(action)
                
                # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                reward = self.processReward(reward)
                nextState = self.processState(nextState)
                self.replayMemory.push(state, action, reward, nextState, done)

                # Update the RL state
                state = nextState


    def learning(self):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.
        
        INPUTS: /
        
        OUTPUTS: - loss: Loss of the learning procedure.
        """
        
        # Check that the replay memory is filled enough
        if (len(self.replayMemory) >= self.batchSize):

            # Sample a batch of experiences from the replay memory
            batch = self.dataLoaderIter.next()
            state = batch[0].float().to(self.device)
            action = batch[1].long().to(self.device)
            reward = batch[2].float().to(self.device)
            nextState = batch[3].float().to(self.device)
            done = batch[4].float().to(self.device)

            # Compute the current Q values returned by the policy network
            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                expectedQValues = reward + self.gamma * nextQValues * (1 - done)

            # Compute the loss (typically Huber or MSE loss)
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()

        
    def training(self, trainingEnv, numberOfEpisodes, verbose=True, rendering=False, plotTraining=True):
        """
        GOAL: Train the RL agent by interacting with the RL environment.
        
        INPUTS: - trainingEnv: Training RL environment.
                - numberOfEpisodes: Number of episodes for the training phase.
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the environment rendering.
                - plotTraining: Enable the plotting of the training results.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Initialization of several variables for storing the results
        mainPerformance = []
        riskPerformance = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Fill the replay memory with a number of random experiences
            self.fillReplayMemory(trainingEnv)

            # Training phase for the number of episodes specified as parameter
            for episode in range(numberOfEpisodes):
                
                # Set the initial RL variables
                state = self.processState(trainingEnv.reset())
                done = 0

                # Interact with the training environment until termination
                while done == 0:

                    # Choose an action according to the RL policy and the current RL state
                    action = self.chooseActionEpsilonGreedy(state, self.epsilonValue(self.steps))
                    
                    # Interact with the environment with the chosen action
                    nextState, reward, done, _ = trainingEnv.step(action)
                    
                    # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                    reward = self.processReward(reward)
                    nextState = self.processState(nextState)
                    self.replayMemory.push(state, action, reward, nextState, done)

                    # Execute the learning procedure of the RL algorithm
                    if self.steps % self.learningUpdatePeriod == 0:
                        self.dataLoader = DataLoader(dataset=self.replayMemory, batch_size=self.batchSize, shuffle=True)
                        self.dataLoaderIter = iter(self.dataLoader)
                        self.learning()

                    # If required, update the target deep neural network (update frequency)
                    self.updateTargetNetwork()

                    # Update the RL state
                    state = nextState

                # Store and report the performance of the RL policy (both expected performance and risk)
                if episode % 10 == 0:
                    _, mainScore, riskScore = self.testing(trainingEnv, False, False)
                    mainPerformance.append([episode, mainScore])
                    riskPerformance.append([episode, riskScore])
                    self.writer.add_scalar("Main performance", mainScore, episode)
                    self.writer.add_scalar("Risk performance", riskScore, episode)
                    mainPerformanceDataframe = pd.DataFrame(mainPerformance, columns=['Episode', 'Score'])
                    riskPerformanceDataframe = pd.DataFrame(riskPerformance, columns=['Episode', 'Score'])
                    mainPerformanceDataframe.to_csv(''.join([self.experimentFolder, 'MainResults.csv']))
                    riskPerformanceDataframe.to_csv(''.join([self.experimentFolder, 'RiskResults.csv']))

                # If required, print a training feedback
                if verbose:
                    print("".join(["Episode ", str(episode+1), "/", str(numberOfEpisodes)]), end='\r', flush=True)
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()

        # Assess the algorithm performance on the training environment
        trainingEnv, mainScore, riskScore = self.testing(trainingEnv, verbose, rendering)

        # Store the testing results into a csv file
        mainPerformanceDataframe = pd.DataFrame(mainPerformance, columns=['Episode', 'Score'])
        riskPerformanceDataframe = pd.DataFrame(riskPerformance, columns=['Episode', 'Score'])
        mainPerformanceDataframe.to_csv(''.join([self.experimentFolder, 'MainResults.csv']))
        riskPerformanceDataframe.to_csv(''.join([self.experimentFolder, 'RiskResults.csv']))

        # If required, plot the training results
        if plotTraining:
            plt.figure()
            mainPerformanceDataframe.plot(x='Episode', y='Score')
            plt.xlabel('Episode')
            plt.ylabel('Main score')
            plt.savefig(''.join([self.experimentFolder, 'MainScore.png']))
            plt.show()
            plt.figure()
            riskPerformanceDataframe.plot(x='Episode', y='Score')
            plt.xlabel('Episode')
            plt.ylabel('Risk score')
            plt.savefig(''.join([self.experimentFolder, 'RiskScore.png']))
            plt.show()

        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv


    def testing(self, testingEnv, verbose=True, rendering=True):
        """
        GOAL: Test the RL agent trained on the RL environment provided.
        
        INPUTS: - testingEnv: Testing RL environment.
                - verbose: Enable the printing of the testing performance.
                - rendering: Enable the rendering of the RL environment.
        
        OUTPUTS: - testingEnv: Testing RL environment.
                 - testingScore: Score associated with the testing phase.
                 - riskScore: Score in terms of riskiness.
        """

        # Initialization of some RL variables
        state = self.processState(testingEnv.reset())
        done = 0

        # Initialization of some variables tracking the RL agent performance
        testingScore = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action = self.chooseActionEpsilonGreedy(state, self.epsilonTest)

            # If required, show the environment rendering
            if rendering:
                testingEnv.render()
                self.chooseAction(state, True)
                
            # Interact with the environment with the chosen action
            nextState, reward, done, info = testingEnv.step(action)
                
            # Process the RL variables retrieved
            state = self.processState(nextState)
            reward = self.processReward(reward)

            # Continuous tracking of the training performance
            testingScore += reward

        # If required, print the testing performance
        if verbose:
            print("".join(["Test environment: main score = ", str(testingScore), "   /   risk score = ", str(info['riskSensitivePolicy'])]))

        return testingEnv, testingScore, info['riskSensitivePolicy']

        
    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, by saving the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load the RL policy, by loading the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).
        
        INPUTS: /
        
        OUTPUTS: /
        """

        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(1000000)])
        plt.xlabel("Steps")
        plt.ylabel("Epsilon")
        plt.show()
        