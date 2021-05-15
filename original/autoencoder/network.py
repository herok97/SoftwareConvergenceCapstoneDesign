import torch.nn as nn
import torch
from torch.autograd import Variable
from lstmcell import StackedLSTMCell

class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, frame_features):
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)
        return (h_last, c_last)

class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden

        out_features = []
        for i in range(seq_len):
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)

        return out_features


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)


    def forward(self, features):
        seq_len = features.size(0)
        h, c = self.e_lstm(features)
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return decoded_features


if __name__ == '__main__':
    pass
