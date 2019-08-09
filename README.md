# SpaceInvadersAgentRL
Implementation of DQN algorithm on the atari game Space Invaders

# Network Class:
CNN (convolutional neural network) used as an approximator of the Q-values given 4 consecutive frames of the game as input.
The CNN outputs Q-values [a.k.a for each action a outputs Q(s,a)].
The network uses a veriant of SGD as an optimizer and relu as an activation function and loss is calculated from MSE.

# Agent Class:
RL agent which capable of storing experience replay in memory and uses it to the learning process using Q-learning with Q-values being approximated using neural network.

# Main:
1. Create an environment and an agent
2. Initialize memory with random transitions
3. Train the agent to play the game using the neural network
