import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2, hidden_size=16):
        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_outputs)

    def forward(self, x, hidden=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        # x [batch_size, sequence_length, num_inputs]
        out, hidden = self.lstm(x, hidden)

        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue
