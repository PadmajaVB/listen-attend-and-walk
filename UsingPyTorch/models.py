from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
        self.gru = nn.GRU(hidden_size, self.hidden_size2, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # hidden = hidden.view(1, 1, self.hidden_size)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size // 2))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    # hidden = 256, output_size = 2925
    def __init__(self, hidden_size, world_state_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.world_state_size = world_state_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dense = nn.Linear(self.world_state_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dense(input)
        embedded = embedded.view(1, 1, -1)
        embedded = self.dropout(embedded)
        # print("embedded[0] = ", embedded[0])  # (1,128)
        # print("hidden[0] = ", hidden[0])  # (1, 128)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  # attn_weights (1,46)  encoder_outputs (46, 128)
        # print("attn_weights = ", attn_weights)
        # print("encoder_outputs = ", encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        # print("output = ", output)  # (1, 4)
        # print("hidden = ", hidden)  # (1, 1, 128)
        # print("attn_weights = ", attn_weights)  # (1, 46)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
