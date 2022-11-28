# Risk-Sensitive Policy with Distributional Reinforcemet Learning

Experimental code supporting the results presented in the scientific research paper:
> Thibaut Théate and Damien Ernst. "Risk-Sensitive Policy with Distributional Reinforcemet Learning." (2022).
> [[arxiv]](https://arxiv.org/abs/)



# Dependencies

The dependencies are listed in the text file "requirements.txt":
* Python 3.7.7
* Pytorch
* Tensorboard
* Gym
* Umnn
* Numpy
* Pandas
* Matplotlib
* Scipy
* Tqdm



# Usage

Training and testing a chosen (risk-sensitive) RL algorithm for the control problem of a chosen environment is performed by running the following command:

```bash
python main.py -algorithm ALGORITHM -environment ENVIRONMENT
```

with:
* ALGORITHM being the name of the algorithm (by default UMDQN_C),
* ENVIRONMENT being the name of the environment (by default RiskyEnvironment).

The RL algorithms supported are:
* DQN,
* UMDQN_C.

The benchmark environments supported are:
* RiskyRewards,
* RiskyTransitions,
* RiskyEnvironment.

The number of episodes for training the DRL algorithm may also be specified by the user through the argument "-episodes". The parameters of the DRL algorithms can be set with the argument "-parameters" and by providing the name of the .json file containing these parameters within the "Parameters" folder.

For more advanced tests and manipulations, please directly refer to the code.



# Citation

If you make use of this experimental code, please cite the associated research paper:

```
@inproceedings{Théate2022,
  title={Risk-Sensitive Policy with Distributional Reinforcemet Learning},
  author={Thibaut Théate and Damien Ernst},
  year={2022}
}
```