import gym
import numpy as np
import Agent

NUM_ACTIONS = 6
NUM_EPISODES = 50
BATCH_SIZE = 32


def init_memory(env, agent):
    while agent.memoryCounter < agent.memorySize:
        current_state = env.reset()
        done = False
        while not done:
            # 0 = no-op, 1 = fire, 2 = right, 3 = left, 4 = right&fire, 5 = left&fire
            currentAction = env.action_space.sample()
            nextState, reward, done, info = env.step(currentAction)
            agent.storeTransition(np.mean(current_state[15:200, 30:125], axis=2), currentAction, reward,
                                  np.mean(nextState[15:200, 30:125], axis=2))
            current_state = nextState


def trainSession(env, agent):
    totalRewards = []

    for episode in range(NUM_EPISODES):
        print('Episode', episode + 1, 'started')
        done = False
        currentState = env.reset()
        # Truncate the sides (left,right,bottom,up) because they are irrelevant to the computation process.
        truncatedState = currentState[15:200, 30:125]
        framesStack = [np.sum(truncatedState, axis=2)]
        totalReward = 0
        currentAction = 0
        while not done:
            if len(framesStack) == 4:   # else we will use our last action...
                currentAction = agent.chooseAction(framesStack)
                framesStack = []
            nextState, reward, done, _ = env.step(currentAction)
            truncatedNextState = nextState[15:200, 30:125]
            totalReward += reward
            # Truncate the sides (left,right,bottom,up) because they are irrelevant to the computation process.
            framesStack.append(np.sum(truncatedNextState, axis=2))
            agent.storeTransition(np.mean(truncatedState, axis=2),
                                  currentAction,
                                  reward,
                                  np.mean(truncatedNextState, axis=2))
            truncatedState = truncatedNextState
            agent.learn(BATCH_SIZE)

        totalRewards.append(totalReward)
        print('Episode', episode + 1, 'ended with total reward: ', totalReward)

    return totalRewards


if __name__ == '__main__':
    ''' 
        1. Create an environment
        2. Create an agent
        3. Initialize memory with random transitions
        4. Train the agent to play the game
    '''
    env = gym.make('SpaceInvaders-v0')
    agent = Agent(stepSize=0.003, discountFactor=0.95, explorationRate=1.0,
                  explorationEnd=0.03, memorySize=5000, actionSpace=[i for i in range(NUM_ACTIONS)])

    print('Initialization phase started!')
    init_memory(env, agent)
    print('Initialization phase ended!')
    print('Training phase started!')
    trainSession(env, agent)
    print('Training phase ended!')


# TODO: save the weights that had been learnt so far and use them to test the agent on the enviroment without any learning process is done.
