import torch.nn as nn
import torch
from torch.autograd import Variable
from .lstmcell import StackedLSTMCell

# Summarizer.py

class sLSTM(nn.Module):
    # input feature vector : (seq_len, 1024)
    def __init__(self, input_size, s_hidden, num_layer=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, s_hidden, num_layer, bidirectional=True, dropout=0.5)
        self.out = nn.Sequential(
            nn.Linear(s_hidden * 2, 1),
            nn.Sigmoid())# SUM-GAN_sup 에서는 softmax 사용

    def forward(self, features):
        """
         self.lstm.flatten_parameters() 호출 이유
            RNN 모듈을 생성하면 각 레이어/방향마다
            w_ih, w_hh, b_ih, b_hh 텐서를 생성해 모듈의 파라미터로 등록되는데
            이 과정에서 각 weight이 메모리 상에서 연속되게 위치함이 보장되지 않는다. => 성능 저하
            가장 적절한 위치가 forward의 처음임
        """
        self.lstm.flatten_parameters()


        # 여기서 좌변의 features는 lstm의 output 인데 bidirectional 이므로 2048가 됨
        # [seq_len, 2048] 를 input으로 넣으면
        # LSTM은 hidden state와 Cell state가 존재함 이를 h_n, c_n 으로 할당하였고
        # h_n 과 c_n 은 둘 다 shape of (batch_size, num_layers, hidden_size) 를 가짐
        # [seq_len, 1, 2048]
        features, (h_n, c_n) = self.lstm(features)
        # [seq_len, 2048] => [seq_len, 1, 2048] => [seq_len, 1, 1]
        scores = self.out(features.squeeze(1))
        return scores

class eLSTM(nn.Module):
    def __init__(self, input_size, e_hidden, d_hidden):
        super().__init__()
        # input 1024 받아서 2048 unit의 hidden state

        self.lstm = nn.LSTM(input_size, e_hidden, 2, dropout=0.5)

        # VAE에서 사용할 확률분호의 평균과 분산.
        # 분산은 이후에 log를 취하던데 log_var 를 KL_loss를 구할 때 사용하니까 그런듯
        # 근데 얘네 왜 forward 에서 사용안할까?
        self.linear_mu = nn.Linear(e_hidden, e_hidden)
        self.linear_var = nn.Linear(e_hidden, e_hidden)

    # 인풋으로 [seq_len, 1, 1024]이 들어옴
    def forward(self, frame_features):

        #위에서 설명했듯이 메모리 직렬화
        self.lstm.flatten_parameters()

        # frame_features [seq_len, 1, 1024]
        # h_last [num_layer=2, 1, 2048]
        # c_last [num_layer=2, 1, 2048]
        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size, e_hidden, d_hidden):
        super().__init__()

        # input param 순서가 조금 다름.
        self.lstm_cell = StackedLSTMCell(2, input_size, d_hidden)
        self.out = nn.Linear(d_hidden, input_size)

    def forward(self, seq_len, init_hidden):    #init_hidden은 eLSTM으로부터 받아오는 것 같음.
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1) # hidden state의 batch size 1
        hidden_size = init_hidden[0].size(2) # hidden state의 hideen_size 2048

        # x [1, 1024] shape의 0 tensor를 할당   # input size
        x = Variable(torch.zeros(batch_size, 1024)).cuda()

        # h, c last state of eLSTM
        h, c = init_hidden

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))

            x = self.out(last_h)
            out_features.append(x)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size, e_hidden, d_hidden):
        super().__init__()
        self.e_lstm = eLSTM(input_size, e_hidden, d_hidden)
        self.d_lstm = dLSTM(input_size, e_hidden, d_hidden)

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1) # std.size() : [2, 2048]

        # 내가 수정한 코드 GPU에 한 번에 올리기
        # epsilon = Variable(torch.randn(std.size())).cuda()
        epsilon = Variable(torch.randn(std.size(), device=torch.device('cuda:0')))

        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h: [2=num_layers, 1, hidden_size]
            decoded_features: [seq_len, 1, 2048]
        """
        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        # [num_layers, hidden_size]
        h = h.squeeze(1)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)

        # 의문: 여기서 왜 softplus 사용하는지 설명 안해줌!?
        # 내가 수정한 코드 / softplus 없앴음
        # h_log_variance = torch.log(self.e_lstm.linear_var(h))
        # 내가 수정한 코드 / 여기서 미리 로그를 씌우지 말았어야죠 앞에다가 softplus를 할게 아니라.
        h_log_variance = self.e_lstm.linear_var(h)

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)
        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))
        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, s_hidden, e_hidden, d_hidden):
        super().__init__()
        self.s_lstm = sLSTM(input_size, s_hidden, 2)

        self.vae = VAE(input_size, e_hidden, d_hidden)

    def forward(self, image_features, uniform=False):
        # Apply weights
        if not uniform:
            # [seq_len, 1]
            scores = self.s_lstm(image_features)
            # [seq_len, 1, 2048] * [seq_len, 1, 1]
            weighted_features = image_features * scores.view(-1, 1, 1)
        else:
            seq_len = image_features.size(0)
            scores = torch.rand(seq_len, 1, 1, device=torch.device('cuda:0'))
            # 내가 수정한 코드 / uniform distribute 에서 추출
            # 의문: 내 생각에는 여기에 0.3? 씩이라도 붙여야 할 것 같은데.. => uniform distribution 으로 붙임
            weighted_features = image_features * scores
        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features


if __name__ == '__main__':
    pass
