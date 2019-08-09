import torch as T
import numpy as np
import Network

NUM_ACTIONS = 6


class Agent(object):
    '''
        Agent Class:
            An agent which capable of learning Space Invaders using neural networks and some memory of his experiences.
    '''
    def __init__(self, learningRate=0.003, discountFactor=0.95, explorationRate=1.0,
                 explorationEnd=0.03, memorySize=5000, actionSpace=None):
        if actionSpace is None:
            actionSpace = [i for i in range(NUM_ACTIONS)]
        self.DISCOUNT_FACTOR = discountFactor
        self.EXPLORATION_RATE = explorationRate
        self.EXPLORATION_END = explorationEnd
        self.memorySize = memorySize
        self.actionSpace = actionSpace
        self.steps = 0  # till it gets to 300 we don't want to act greedy.
        self.memory = []
        self.memoryCounter = 0
        self.Q_predictions = Network(learningRate) # Agent predictions (Q-values of states)

    def storeTransition(self, currentState, action, reward, nextState):
        if self.memoryCounter < self.memorySize:
            self.memory.append([currentState, action, reward, nextState])
        else:
            self.memory[self.memoryCounter % self.memorySize] = [currentState, action, reward, nextState]
        self.memoryCounter += 1

    def chooseAction(self, state):
        # state is stack of frames
        actions = self.Q_predictions.forward(state)
        if np.random.random() > self.EXPLORATION_RATE:
            action = T.argmax(actions[1]).item()        # Act greedly
        else:
            action = np.random.choice(self.actionSpace) # Act randomly
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_predictions.optimizer.zero_grad() # we want to learn differently for each batch

        # select a random sub-sample of the memory in size batch_size
        if self.memoryCounter + batch_size < self.memorySize:
            memoryStart = int(np.random.choice(range(self.memoryCounter)))
        else:
            memoryStart = int(np.random.choice(range(self.memorySize - batch_size - 1)))
        miniBatch = self.memory[memoryStart : memoryStart + batch_size]
        memory = np.array(miniBatch) # convert to numpy array so we could operate on it easily

        # convert to list because memory is an array of numpy objects
        # memory[rows=for each stored transition : column=positions in transitions (curr,action,reward,next)][pixels]
        Qpred = self.Q_predictions.forward(list(memory[:, 0][:]))
        Qnext = self.Q_predictions.forward(list(memory[:, 3][:]))

        # Update the target values for the action which maximizes the Q-value.
        # This is done in order to update backward the weights of the neural network.
        action = T.argmax(Qnext, dim=1)
        rewards = T.Tensor(list(memory[:, 2]))
        Qtarget = Qpred
        maxQnextValue = T.max(Qnext[1])
        Qtarget[np.arange(batch_size), action] = rewards + self.DISCOUNT_FACTOR * maxQnextValue
        # Calculate loss between Qtarget and Qpred
        loss = self.Q_predictions.loss(Qtarget, Qpred)
        loss.backward()
        self.Q_predictions.optimizer.step()

        # Update exploration rate
        if self.steps > 300:
            if self.EXPLORATION_RATE - 0.00005 > self.EXPLORATION_END:
                self.EXPLORATION_RATE -= 0.00005
            else:
                self.EXPLORATION_RATE = self.EXPLORATION_END
