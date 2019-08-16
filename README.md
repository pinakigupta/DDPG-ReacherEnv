# Project 2: (Unity) Reacher Enviornment

<img src="https://camo.githubusercontent.com/7ad5cdff66f7229c4e9822882b3c8e57960dca4e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4a756e652f35623165613737385f726561636865722f726561636865722e676966">

# Introduction
    Set-up: Double-jointed arm which can move to target locations.
    Goal: The agents must move its hand to the goal location, and keep it there.
    Agents: The environment contains 10 agent linked to a single Brain.
    Agent Reward Function (independent):
    +0.1 Each step agent's hand is in goal location.
    Brains: One Brain with the following observation/action space.
    Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
    Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
    Visual Observations: None.
    Reset Parameters: Five
    goal_size: radius of the goal zone
    Default: 5
    Recommended Minimum: 1
    Recommended Maximum: 10
    goal_speed: speed of the goal zone around the arm (in radians)
    Default: 1
    Recommended Minimum: 0.2
    Recommended Maximum: 4
    gravity
    Default: 9.81
    Recommended Minimum: 4
    Recommended Maximum: 20
    deviation: Magnitude of sinusoidal (cosine) deviation of the goal along the vertical dimension
    Default: 0
    Recommended Minimum: 0
    Recommended Maximum: 5
    deviation_freq: Frequency of the cosine deviation of the goal along the vertical dimension
    Default: 0
    Recommended Minimum: 0
    Recommended Maximum: 3
    **Benchmark Mean Reward: 30**
    


### Introduction:
- To run the project just execute the <b>Continuous_Control.py</b> file.
- There is also an Continuous_Control.ipynb file for jupyter notebook execution.
- The <b>MultiEnvCheckPt/*checkpoint.pth</b> has the checkpointed actor and critic models
- environment.yml has the list of dependencies. The list includes few packages which are not used.




## snapshot:
- The task solved here refers to a continuous control problem where the agent must be able to reach and go along with a moving ball controlling its arms.
- It's a continuous problem because the action has a continuous value and the agent must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The environment provides 2 versions, one with just 1 agent and another one with 20 agents working in parallel.
- For both versions the goal is to get an average score of +30 over 100 consecutive episodes (for the second version, the average score of all agents must be +30).
- For the checkpointed solution the second version of the environment has been used. This turns out to converge much faster due to the collective experience from multiple environments
- For the solution a Deep Deterministic Policy Gradients algorithm has been used.



### Code Description

1. Continuous_Control.ipynb - Main module containing 1)loading of the helper modules 2) loading of the DQN agent helper module 3)training the DQN agent 4)plotting results and 5) checkpointing the model parameters.
2. model.py - loads pytorch module and derives a custom NN model for this problem
3. ddpg_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a DDPG agent 3) Contains experience replay buffer from which the DQN draws sample 4)Also processes multiple environment parallely 





### Important hyperparameters:
1. Continuous_Control.ipynb - Main Module contains most of the hyper parameters for training the DDPG agent 
  - n_episodes Size: 1000     # Maximum number of episodes for which training will proceed
  - checkpoint_score: 30     # if the score is greater than this threshold, network is checkpointed and training is finished. 



2. ddpg_agent.py - contains most of the agent parameters
  - Learning Rate: 1e-4 (in both DNN actor/critic) # learning rate 
  - Batch Size: 128     # minibatch size
  - Replay Buffer: 1e5  # replay buffer size
  - Gamma: 0.99         # discount factor
  - Tau: 1e-3           # for soft update of target parameters
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.) # Noise use to introduce entropy in the system to explore more

3. model.py contains the NN architecture and associated parameters
- For the neural models:    
  - Actor    
    - Hidden: (input, 256)  - ReLU
    - Hidden: (256, 128)    - ReLU
    - Output: (128, 4)      - TanH

  - Critic
    - Hidden: (input, 256)              - ReLU
    - Hidden: (256 + action_size, 128)  - ReLU
    - Hidden: (128, 128)  - ReLU
    - Output: (128, 1)                  - Linear
