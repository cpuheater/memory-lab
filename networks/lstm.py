import torch.nn as nn
import torch
from torch import Tensor
import math
import torch.nn.functional as F

class LSTMWithLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(LSTMWithLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_t, _) = self.lstm(x)
        return self.fc(h_t)


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        bs, seq_length, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_dim).to(x.device),
                        torch.zeros(bs, self.hidden_dim).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_dim
        for t in range(seq_length):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input gate
                torch.sigmoid(gates[:, HS:HS*2]), # forget gate
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    

class CustomLSTM_EXP1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.norm = nn.LayerNorm(hidden_dim)
        self.is_init = False
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        bs, seq_length, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_dim).to(x.device),
                        torch.zeros(bs, self.hidden_dim).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_dim
        for t in range(seq_length):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input gate
                torch.sigmoid(gates[:, HS:HS*2]), # forget gate
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output gate
            )
            c_t = f_t * c_t + i_t * g_t    
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class WithLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn):
        super(WithLinear, self).__init__()
        self.rnn = rnn
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        outputs, (h_t, c_t) = self.rnn(x)
        out = self.fc(outputs)
        #if out.dim() > 2:
        #    out = out.squeeze(0)
        return out


class WithLinearAndEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn):
        super(WithLinearAndEmbed, self).__init__()
        self.rnn = rnn
        self.embed = nn.Embedding.from_pretrained(torch.eye(input_dim), freeze=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = torch.squeeze(x)
        outputs, (h_t, c_t) = self.rnn(x)
        out = self.fc(outputs)
        #if out.dim() > 2:
        #    out = out.squeeze(0)
        return out
    
    