from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional  as F

use_cuda = torch.cuda.is_available()

MAX_LENGTH = 46


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectionality = False):
        super(EncoderRNN, self).__init__()
        if bidirectionality is True:
            self.hidden_size = hidden_size
            self.hidden_size2 = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        " Set bi-directionality = True "
        self.lstm = nn.LSTM(hidden_size, self.hidden_size2, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        # hidden = hidden.view(1, 1, self.hidden_size)
        return output, hidden

    def initHidden(self):
        h0 = Variable(torch.zeros(2, 1, self.hidden_size // 2))
        c0 = Variable(torch.zeros(2, 1, self.hidden_size // 2))
        result = (h0, c0)
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    # hidden = 256, output_size = 2925
    def __init__(self, input_size, hidden_size, world_state_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.world_state_size = world_state_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.input_hidden_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.transform_beta = nn.Linear(self.hidden_size, 1)
        self.decoder_input = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        self.dense = nn.Linear(self.world_state_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, input, world_state, hidden, encoder_outputs):
        # encoder_outputs = encoder_outputs.unsqueeze(0)  # (1, 46, 128)
        embed = self.embedding(input)  # Padding?
        embedded = Variable(torch.zeros(self.max_length, self.hidden_size))

        """Error : copy.copy"""
        for idx, e in enumerate(embed):
            embedded[idx] = e
        # embedded = embedded.view(1, embedded.shape[0], embedded.shape[1])  # (1, 46, 128)

        scope_attr = self.input_hidden_combine(torch.cat((embedded, encoder_outputs), 1))  # (46, 128)
        beta_inprocess = scope_attr + hidden[0][0]  # hidden[0] = (1, 128),  (46, 128) + (1, 128) = (46, 128)
        beta = F.tanh(beta_inprocess)  # (46, 128)
        beta = self.transform_beta(beta)  # (46, 1)

        attn_weights1 = F.softmax(beta, dim=0)  # alpha  # (46, 1)
        attn_weights = torch.t(attn_weights1)  # Transpose (1, 46)
        zt = torch.bmm(attn_weights.unsqueeze(0), scope_attr.unsqueeze(0))  # zt -- context vector (1, 1, 128)

        world_state = world_state.view(1, -1)
        world_state = self.dense(world_state)

        output = torch.cat((zt[0], world_state, hidden[0][0]), 1)  # (1, 128*3)
        output = self.decoder_input(output).unsqueeze(0)  # (1, 1, 128)
        output, hidden = self.lstm(output, hidden)  # output --> st

        output_ctx_combine = self.linear(torch.cat((output[0], zt[0]), 1))  # (1, 256) --> (1, 128)
        qt = self.out(world_state + output_ctx_combine)

        output = F.log_softmax(qt, dim=1)  # (1, 4)
        return output, hidden, attn_weights1

    def initHidden(self):
        h0 = Variable(torch.zeros(1, 1, self.hidden_size))
        c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        result = (h0, c0)
        if use_cuda:
            return result.cuda()
        else:
            return result
