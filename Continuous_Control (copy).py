# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
#from IPython import get_ipython

#%% [markdown]
# # Continuous Control
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

#%%
#get_ipython().system('pip -q install ./python')

#%% [markdown]
# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.  
# 
# Please select one of the two options below for loading the environment.

#%%
from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64',no_graphics=True)

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')

#%% [markdown]
# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

#%%
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
for brains in env.brain_names:
    print(brains)

#%% [markdown]
# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

#%%
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

#%% [markdown]
# ### 3. Loading DDPG Agent
# 

#%%
from importlib import reload 
from collections import deque
import ddpg_agent
reload(ddpg_agent)
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE is ", DEVICE)
#from workspace_utils import active_session


#%%
def play(env = None):
    for i in range(3):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break 

#%%
def train(env = None, n_episodes=200, agent = None, 
         checkpoint_score = 25, breakpoint_score = 30, filename_prefix = ""):
    
    
    scores_deque = deque(maxlen=100)
    scores = []
    goal_steps = []
    goal_rewards = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        agent.reset()
        score = 0
        modified_score = 0
        goal_steps.clear()
        goal_rewards.clear()
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            if reward > 0.0:
                #goal_steps.append(t)
                goal_rewards.append(reward)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        fname = filename_prefix

        print('\rEpisode {}\tAverage Score: {:.2f}\tscore: {:.2f}'.
              format(i_episode, np.mean(scores_deque), score), end="")
        #print("\ngoal_steps ", goal_steps)
        #print("goal_rewards ", goal_rewards)
        #print("\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=checkpoint_score:
                fname += str(i_episode)
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                torch.save(agent.qnetwork_local.state_dict(), fname + 'checkpoint.pth')
        if np.mean(scores_deque)>=breakpoint_score:
            fname += str(i_episode)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.qnetwork_local.state_dict(),  fname + 'checkpoint.pth')
            break   
    
    return scores




#%%

agent = ddpg_agent.Agent(state_size=state_size, 
                         action_size=action_size, 
                         random_seed=0)


#%%
rr_scores = train( env = env,
                   agent = agent) # random replay training


#%%
import matplotlib.pyplot as plt

def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


#%%
plot_scores(rr_scores) # random replay scores

#%% 
# When finished, you can close the environment.
# load the weights from file
play(env = env)


            

#%%
env.close()


