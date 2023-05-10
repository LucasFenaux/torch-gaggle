import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2, hidden_size=16):
        super(DQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively.
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers.

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        # The variable x denotes the input to the network.
        # The function returns the q value for the given input.

        x = x.view(-1, self.num_inputs)
        x = F.sigmoid(self.fc1(x))
        qvalue = self.fc2(x)  # wouldn't usually do a second sigmoid but leap does it so we have to
        return qvalue