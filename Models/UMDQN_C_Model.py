# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.FeedforwardDNN import FeedForwardDNN
from Models.MonotonicNN import MonotonicNN



###############################################################################
############################ Class UMDQN_C_Model ##############################
###############################################################################

class UMDQN_C_Model(nn.Module):
    """
    GOAL: Implementing the DL model for the UMDQN-C distributional RL algorithm.
    
    VARIABLES:  - stateEmbeddingDNN: State embedding part of the Deep Neural Network.
                - UMNN: UMNN part of the Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
                - getDerivative: Get the derivative internally computed by the UMNN.
                - getExpectation: Get the expectation of the PDF internally computed by the UMNN.
    """

    def __init__(self, numberOfInputs, numberOfOutputs,
                 structureDNN, structureUMNN, stateEmbedding,
                 numberOfSteps, device='cpu'):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structureDNN: Structure of the feedforward DNN for state embedding.
                - structureUMNN: Structure of the UMNN for distribution representation.
                - stateEmbedding: Dimension of the state embedding.
                - numberOfSteps: Number of integration steps for the UMNN.
                - device: Hardware device (CPU or GPU).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(UMDQN_C_Model, self).__init__()

        # Initialization of the Deep Neural Network
        self.stateEmbeddingDNN = FeedForwardDNN(numberOfInputs, stateEmbedding, structureDNN)
        self.UMNN = MonotonicNN(stateEmbedding+1, structureUMNN, numberOfSteps, numberOfOutputs, device)

    
    def forward(self, state, q):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - state: RL state.
                - q: Samples of potential returns.
        
        OUTPUTS: - output: Output of the Deep Neural Network.
        """
        
        # State embedding part of the Deep Neural Network
        batchSize = state.size(0)
        x = self.stateEmbeddingDNN(state)
        x = x.repeat(1, int(len(q)/len(state))).view(-1, x.size(1))

        # UMNNN part of the Deep Neural Network
        x = self.UMNN(q, x)

        # Sigmoid activation function + appropriate format
        x = torch.sigmoid(x)
        return torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)


    def getDerivative(self, state, q):
        """
        GOAL: Get the derivative internally computed by the UMNN.
        
        INPUTS: - state: RL state.
                - q: Samples of potential returns.
        
        OUTPUTS: - output: Derivative internally computed by the UMNN.
        """

        # State embedding part of the Deep Neural Network
        batchSize = state.size(0)
        x = self.stateEmbeddingDNN(state)
        x = x.repeat(1, int(len(q)/len(state))).view(-1, x.size(1))

        # Computation of both PDF and CDF
        pdf = self.UMNN(q, x, only_derivative=True)
        cdf = self.UMNN(q, x, only_derivative=False)

        # Correction of the sigmoid + appropriate format
        x = torch.sigmoid(cdf)
        x = x * (1 - x) * pdf
        return torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)


    def getExpectation(self, state, minReturn, maxReturn, numberOfPoints):
        """
        GOAL: Get the expectation of the PDF internally computed by the UMNN.
        
        INPUTS: - state: RL state.
                - minReturn: Minimum return.
                - maxReturn: Maximum return.
                - numberOfPoints: Number of points for the computations (accuracy).
        
        OUTPUTS: - expectation: Expectation computed.
        """

        # State embedding part of the Deep Neural Network
        state = self.stateEmbeddingDNN(state)

        # Computation of the expectation of the PDF internally computed by the UMNN
        expectation = self.UMNN.expectation(state, lambda x: x, lambda x: torch.sigmoid(x)*(1-torch.sigmoid(x)), minReturn, maxReturn, numberOfPoints)
        return expectation

