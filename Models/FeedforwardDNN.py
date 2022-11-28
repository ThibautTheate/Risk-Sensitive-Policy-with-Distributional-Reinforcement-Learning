# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102



###############################################################################
############################ Class FeedForwardDNN #############################
###############################################################################

class FeedForwardDNN(nn.Module):
    """
    GOAL: Implementing a classical feedforward DNN using Pytorch.
    
    VARIABLES:  - network: Feedforward DNN.
                                
    METHODS:    - __init__: Initialization of the feedforward DNN.
                - forward: Forward pass of the feedforward DNN.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, structure):
        """
        GOAL: Defining and initializing the feedforward DNN.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structure: Structure of the feedforward DNN (hidden layers).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(FeedForwardDNN, self).__init__()

        # Initialization of the FeedForward DNN
        self.network = []
        structure = [numberOfInputs] + structure + [numberOfOutputs]
        for inFeature, outFeature in zip(structure, structure[1:]):
            self.network.extend([
                nn.Linear(inFeature, outFeature),
                nn.ReLU(),
            ])
        self.network.pop()
        self.network = nn.Sequential(*self.network)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the feedforward DNN.
        
        INPUTS: - x: Input of the feedforward DNN.
        
        OUTPUTS: - y: Output of the feedforward DNN.
        """

        return self.network(x)
