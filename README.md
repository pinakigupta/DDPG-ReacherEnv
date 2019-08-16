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




## The problem:
- The task solved here refers to a continuous control problem where the agent must be able to reach and go along with a moving ball controlling its arms.
- It's a continuous problem because the action has a continuous value and the agent must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The environment provides 2 versions, one with just 1 agent and another one with 20 agents working in parallel.
- For both versions the goal is to get an average score of +30 over 100 consecutive episodes (for the second version, the average score of all agents must be +30).


## The solution:
- For this problem I used an implementation of the Deep Deterministic Policy Gradients algorithm.
- This task brought two big challenges for me: hyperparameters tunning and noise range configuration. After I found the right configuration for these two points the solution worked impressively well. I must say that the noise range configuration is the key for this task. As the action is a continuous value, dealing with noise correctly means more generalization and makes the agent convergence faster and more robust. The other hyperparameters increase the convergence speed but almost never prevent the agent from finding the solution whereas the wrong noise range configuration can easily make the agent unstable and, I risk saying, impossible to converge.
- Another thing to highlight here is how great the approach used in actor critic structures in general is. It really takes the good part of both worlds, value based methods and policy gradient methods, and makes them work together in an impressive way. Especially in this task, the way the actor and critic learn together sharing their experiences really brought to my eyes a revolutionary point of view about how to build machine learning algorithms. It's really worth to take a look.
- For the future, although the actual solution seems pretty good to me, I stil want to check this task with the D4PG algorithm and discover when and where each of the algorithms (DDPG vs. D4PG) have the best performance.


### Code Description

1. Continuous_Control.ipynb - Main module containing 1)loading of the helper modules 2) loading of the DQN agent helper module 3)training the DQN agent 4)plotting results and 5) checkpointing the model parameters.
2. model.py - loads pytroch module and derives a custom NN model for this problem
3. ddpg_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a DDPG agent 3) Contains experience replay buffer from which the DQN draws sample 4)Also processes multiple environment parallely 


### Important Hyperparameters 

1. Continuous_Control.ipynb - Main Module contains most of the hyper parameters for training the DQN agent 
		a. n_episodes. Maximum number of episodes for which training will proceed
		b. max_t. maximum number of steps per episode during training
		c. eps_start, eps_end, eps_decay - During the exploration using an episilon greedy policy is used. The policy starts with eps_start at episode 1 and decays by eps_decay each episode
		until it hits the eps_end floor.
		d. random_replay - If True random sampling of experience buffer is chosen, otherwise the prioritized sampling of experience buffer is chosen
		e. dqn_fc_layer - architecture of the Hidden layers of the Q network. ex. = [ 64 64 32 256] means there are 4 hidden layers of 64, 64, 32 and 256 units, in that order.
		f. checkpoint_score - if the score is greater than this threshold, every 100 episode the network will be checkpointed. This can be set as a score 
		target for a reasonably good agent.
		g. breakpoint_score - if the score is greater than this threshold, network is checkpointed and training is finished. This can be set as a score 
		target for a exceptionally good agent.
2. model.py -   

		BUFFER_SIZE = int(1e5)  # replay buffer size
		BATCH_SIZE = 64         # minibatch size
		GAMMA = 0.99            # discount factor
		TAU = 1e-3              # for soft update of target parameters
		LR = 5e-4               # learning rate 
		UPDATE_EVERY = 4        # how often to update the network





### Important hyperparameters:
1. Continuous_Control.ipynb - Main Module contains most of the hyper parameters for training the DDPG agent 
  - Batch Size: 128     # minibatch size
  - Batch Size: 128     # minibatch size
  - Batch Size: 128     # minibatch size



2. ddpg_agent.py - contains most of the agent parameters
  - Learning Rate: 1e-4 (in both DNN actor/critic) # learning rate 
  - Batch Size: 128     # minibatch size
  - Replay Buffer: 1e5  # replay buffer size
  - Gamma: 0.99         # discount factor
  - Tau: 1e-3           # for soft update of target parameters
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)

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
