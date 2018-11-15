# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
            input: shape=(batch_size, seq_size)
            output: shape=(batch_size, seq_size, output_size)
        """
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size, cuda):
        cuda_wrapper = lambda x: x.cuda() if cuda else x
        if self.model == "lstm":
            return (cuda_wrapper(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))),
                    cuda_wrapper(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))))
        return cuda_wrapper(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

