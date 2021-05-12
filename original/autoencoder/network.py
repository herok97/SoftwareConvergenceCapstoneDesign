import torch.nn as nn
import torch
from torch.autograd import Variable
from lstmcell import StackedLSTMCell

class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):

        self.lstm.flatten_parameters()

        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):    #init_hidden은 eLSTM으로부터 받아오는 것 같음.

        batch_size = init_hidden[0].size(1) # hidden state의 batch size 1
        hidden_size = init_hidden[0].size(2) # hidden state의 hideen_size 2048

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()


    def reparameterize(self, mu, log_variance):
        std = torch.exp(0.5 * log_variance)
        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda()
        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        seq_len = features.size(0)

        h, c = self.e_lstm(features)

        h = h.squeeze(1)

        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        h = self.reparameterize(h_mu, h_log_variance)

        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return decoded_features


if __name__ == '__main__':
    pass
