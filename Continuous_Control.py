#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.  
# 
# Please select one of the two options below for loading the environment.

# In[1]:


from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment
#env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64',no_graphics=True)

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='Reacher_Linux_v2/Reacher.x86_64',no_graphics=True)


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[2]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
for brains in env.brain_names:
    print(brains)


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[3]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_envs = len(env_info.agents)
print('Number of agents:', num_envs)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Loading DDPG Agent
# 

# In[46]:


from importlib import reload 
from collections import deque
import ddpg_agent
reload(ddpg_agent)
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE is ", DEVICE)


# In[47]:


import matplotlib.pyplot as plt

def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


# In[48]:


def train(env = None, n_episodes=1000, agent = None, 
         checkpoint_score = 30, checkpt_folder = ""):
    
    
    scores_deque = deque(maxlen=100)
    scores = []
    goal_steps = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state
        agent.reset()
        score = np.zeros(num_envs)
        modified_score = 0
        goal_steps.clear()
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations     # get the next states
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        print('\rEpisode {}\tAverage Score: {:.2f}\tscore: {:.2f}'.
              format(i_episode, np.mean(scores_deque), scores[-1]), end="")


        if np.mean(scores_deque)>=checkpoint_score:
            checkpt = "Episode" + str(i_episode)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.checkpoint(checkpt)
            break   
    
    return scores


# In[49]:


agent = ddpg_agent.Agent(state_size=state_size, 
                         action_size=action_size, 
                         random_seed=0,
                         num_envs=num_envs,
                         checkpt_folder = "MultiEnvCheckPt")


# In[50]:


rr_scores = train( env = env,
                   agent = agent) # Multiple parallel Env


# In[51]:


plot_scores(rr_scores) # random replay scores


# When finished, you can close the environment.

# In[6]:


env.close()

