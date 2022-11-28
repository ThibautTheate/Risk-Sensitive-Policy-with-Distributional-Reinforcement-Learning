# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse
import importlib

from performance import Performance



###############################################################################
################################ Global variables #############################
###############################################################################

# Supported RL algorithms
algorithms = ['DQN', 'UMDQN_C']

# Supported RL environments
environments = ['RiskyRewards', 'RiskyTransitions', 'RiskyEnvironment']



###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-algorithm", default='UMDQN_C', type=str, help="Name of the RL algorithm")
    parser.add_argument("-environment", default='RiskyEnvironment', type=str, help="Name of the RL environment")
    parser.add_argument("-episodes", default=10000, type=str, help="Number of episodes for training")
    parser.add_argument("-parameters", default='parameters', type=str, help="Name of the JSON parameters file")
    args = parser.parse_args()

    # Checking of the parameters validity
    algorithm = args.algorithm
    environment = args.environment
    episodes = int(args.episodes)
    parameters = args.parameters
    if algorithm not in algorithms:
        print("The algorithm specified is not valid, only the following algorithms are supported:")
        for algo in algorithms:
            print("".join(['- ', algo]))
    if environment not in environments:
        print("The environment specified is not valid, only the following environments are supported:")
        for env in environments:
            print("".join(['- ', env]))
    if parameters == 'parameters':
        parameters = ''.join(['Parameters/parameters_', str(algorithm), '_', str(environment), '.json'])
    
    # Name of the file for saving the RL policy learned
    fileName = 'SavedModels/' + algorithm + '_' + environment
    
    # Initialization of the RL environment
    environmentModule = importlib.import_module(str(environment))
    className = getattr(environmentModule, environment)
    env = className()

    # Determination of the state and action spaces
    observationSpace = env.observation_space.shape[0]
    actionSpace = env.action_space.n

    # Initialization of the DRL algorithm
    algorithmModule = importlib.import_module(str(algorithm))
    className = getattr(algorithmModule, algorithm)
    RLAgent = className(observationSpace, actionSpace, environment, parameters)

    # Training of the RL agent
    RLAgent.training(env, episodes, verbose=True, rendering=False, plotTraining=False)
    
    # Saving of the RL model
    RLAgent.saveModel(fileName)

    # Loading of the RL model
    RLAgent.loadModel(fileName)

    # Testing of the RL agent
    RLAgent.testing(env, verbose=True, rendering=True)

    # Computation of the performance achieved by the learnt RL policy
    alpha = 0.5
    rho = 0.1
    samples = 1000000
    performanceAssessment = Performance(RLAgent, env)
    performanceAssessment.computePerformance(alpha, rho, samples, verbose=True)
