# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np

from tqdm import tqdm
from bisect import bisect_left
from matplotlib import pyplot as plt



###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters for the plotting of the distributions
numberOfSamples = 1000000
bins = 1000
density = True
plotRange = (-2.0, 2.0)



###############################################################################
############################ Class Performance ################################
###############################################################################

class Performance():
    """
    GOAL: Assess quantitatively the performance of a risk-sensitive RL policy.
    
    VARIABLES: - RLAgent: RL agent analysed (algorithm + policy).
               - environment: RL environment studied.
                                
    METHODS: - __init__: Initialization of the performance assessment class.
             - expectation: Estimation of the expectation of a distribution.
             - var: Estimation of the Value at Risk of a distribution.
             - generateSamples: Generation of cumulative reward samples.
             - deriveScoreDistribution: Estimate the cumulative reward distribution.
             - computePerformance: Compute the performance achieved by the RL policy.
    """

    def __init__(self, RLAgent, environment):
        """
        GOAL: Initializing the quantitative performance assessment.
        
        INPUTS: - RLAgent: RL agent analysed (algorithm + policy).
                - environment: RL environment studied.
        
        OUTPUTS: /
        """

        # Initialization of the RL agent together with its environment
        self.RLAgent = RLAgent
        self.environment = environment

    
    def expectation(self, support, PDF):
        """
        GOAL: Compute the Value at Risk for a certain probability, from a
              cumulative distribution function (CDF).
        
        INPUTS: - support: Support of the PDF/CDF (x axis).
                - PDF: Probability distribution function.
        
        OUTPUTS: - mean: Expectation of the distribution
        """

        # Computation of the expectation
        deltaSupport = support[1] - support[0]
        mean = (deltaSupport * PDF * support).mean() * len(PDF)
        return mean

    
    def var(self, support, CDF, probability):
        """
        GOAL: Compute the Value at Risk for a certain probability, from a
              cumulative distribution function (CDF).
        
        INPUTS: - support: Support of the PDF/CDF (x axis).
                - CDF: Cumulative distribution function.
                - probability: Proba associated with the Value at Risk.
        
        OUTPUTS: - risk: Value at risk computed.
        """

        # Computation of the Value at Risk (VaR)
        index = np.argmin(abs(CDF - probability))
        risk = support[index]
        return risk


    def generateSamples(self, numberOfSamples=numberOfSamples):
        """
        GOAL: Generating a set of cumulative rewards based on numerous
              runs of the learnt RL decision-making policy.
        
        INPUTS: - numberOfSamples: Number of cumulative rewards to sample.
        
        OUTPUTS: - scores: Set of cumulative rewards sampled.
        """

        # Initialization of the data structure
        scores = []

        # Loop for generating the set of cumulative rewards
        try:
            for i in tqdm(range(numberOfSamples)):
                _, score, _ = self.RLAgent.testing(self.environment, verbose=False, rendering=False)
                scores.append(score)
        except KeyboardInterrupt:
            print()
            print("WARNING: Samples generation prematurely interrupted...")
            print()

        return scores

    
    def deriveScoreDistribution(self, numberOfSamples=numberOfSamples, plot=False, save=False):
        """
        GOAL: Approximating the distribution of cumulative rewards achieved by
              the learnt RL decision-making policy.
        
        INPUTS: - numberOfSamples: Number of cumulative rewards to sample.
                - plot: Boolean to plot the distribution estimated.
                - save: Boolean to save the distribution estimated.
        
        OUTPUTS: - PDF: PDF of the estimated probability distribution.
                 - CDF: CDF of the estimated probability distribution.
        """

        # Generation of cumulative reward samples
        scores = self.generateSamples(numberOfSamples)

        # Estimation and plotting of the cumulative reward distribution (PDF and CDF)
        ax1 = plt.subplot(2, 1, 1)
        (PDF, support, _) = plt.hist(scores, bins=bins, density=density, range=plotRange, histtype='stepfilled')
        ax1.set_xlabel('Cumulative reward')
        ax1.set_ylabel('PDF')
        ax1.set(xlim=(-1.9, 1.9))
        ax2 = plt.subplot(2, 1, 2)
        (CDF, support, _) = plt.hist(scores, bins=bins, density=density, range=plotRange, histtype='step', cumulative=True)
        ax2.set_xlabel('Cumulative reward')
        ax2.set_ylabel('CDF')
        ax2.set(xlim=(-1.9, 1.9))
        if plot:
            plt.show()
        if save:
            plt.savefig("Figures/Performance/ScoreDistribution.pdf", format='pdf')

        # Return of the both the PDF and CDF of the estimated distribution
        support = [(support[i]+support[i+1])/2 for i in range(len(support)-1)]
        return support, PDF, CDF


    def computePerformance(self, alpha, rho, numberOfSamples=numberOfSamples, verbose=True):
        """
        GOAL: Computing the performance of the learnt RL decision-making policy,
              both in terms of risk and expected return.
        
        INPUTS: - numberOfSamples: Number of cumulative rewards to sample.
        
        OUTPUTS: - Q: Expected return.
                 - R: Risk function.
                 - U: Utility function.
        """

        # Estimate the probability distribution of the cumulative rewards
        support, PDF, CDF = self.deriveScoreDistribution(numberOfSamples, plot=verbose)

        # Computation of the expectation and VaR of the probability distribution
        Q = self.expectation(support, PDF)
        R = self.var(support, CDF, rho)

        # Computation of the utility function, the key performance indicator
        U = alpha * Q + (1-alpha) * R

        # If required, print the results
        print()
        print(''.join(['Expected performance: ', str(Q)]))
        print(''.join(['Risk performance: ', str(R)]))
        print(''.join(['Utility performance: ', str(U)]))
        print()

        # Return of the different performance indicators
        return Q, R, U
