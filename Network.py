import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

NUM_ACTIONS = 6

class Network(nn.Module):
    def __init__(self, learningRate):
        super(Network, self).__init__()
                            # (in_channel, out_channel, kernel)
        # 3 convolutional layers to process the pixels of the images that represent the state
        self.conv_1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.fc_1 = nn.Linear(128 * 19 * 8, 512)
        self.fc_2 = nn.Linear(512, NUM_ACTIONS)  # This outputs distribution over action space
        # Optimizer here is like SGD (Stochastic Gradient Descent) but works much faster,
        # I used very fast learning rate decrament, RMSprop optimizer knows how to deal with this issue.
        # self.parameters() are the inputs and outputs of the neural network
        self.optimizer = optim.RMSprop(self.parameters(), lr=learningRate)
        # loss function -
        # calculate the mean squared error between Q-prediction and Q-target
        # in order to update the weights on the neural network
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    # ReLU activation function is 0 when x<0 and x when x>=0
    def forward(self, states):
        states = T.Tensor(states).to(self.device)
        # these numbers are because we will truncate the frames into the range of [15:200, 30:125] for efficiency
        # -1 to handle any size of stack of frames
        states = states.view(-1, 1, 185, 95)
        states = func.relu(self.conv_1(states))
        states = func.relu(self.conv_2(states))
        states = func.relu(self.conv_3(states))
        # final output out of the convolutional layers
        states = states.view(-1, 128 * 19 * 8)
        states = func.relu(self.fc_1(states))
        actions = self.fc_2(states)
        return actions
