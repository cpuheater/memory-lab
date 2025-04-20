import torch
from .lstm import CustomLSTM, CustomLSTM_EXP1, WithLinear, WithLinearAndEmbed
from .slstm import sLSTMWithLinear
import torch.nn as nn


def select_model(model_type: str, input_dim: torch.Tensor, hidden_dim: torch.Tensor, output_dim: torch.Tensor, use_embed: bool = False):
        linear = WithLinearAndEmbed if use_embed else WithLinear
        match model_type:
            case "CustomLSTM_EXP1":
                rnn = CustomLSTM_EXP1(input_dim = input_dim, hidden_dim = hidden_dim)
                return linear(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, rnn = rnn)
            case "CustomLSTM":
                rnn = CustomLSTM(input_dim = input_dim, hidden_dim = hidden_dim)
                return linear(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, rnn = rnn)
            case "LSTM":
                rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                return linear(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, rnn = rnn)
            case "sLSTM":
                return sLSTMWithLinear(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim)        
