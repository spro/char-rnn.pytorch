# https://github.com/zutotonno/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1,
     dropout = 0, gpu = True, batch_size = 32, chunk_len = 30, learning_rate = 0.001, optimizer = "adam"):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu = gpu
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.optimizer = optimizer

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)
        if self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif self.optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.cuda()

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
             return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

    
    def train(self,inp, target, validation):
        self.zero_grad()
        loss = 0
        hidden = self.init_hidden(self.batch_size)
        if self.cuda:
            if self.model == "gru":
                hidden = hidden.cuda()
            else:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
        for c in range(self.chunk_len):
            output, hidden = self(inp[:, c], hidden)
            loss += self.criterion(output.view(self.batch_size, -1), target[:, c])       
         ### The losses are averaged across observations for each minibatch (see doc CrossEntropyLoss)
        if not validation:
            loss.backward()
            self.optimizer.step()
        currentLoss = loss.item()/ self.chunk_len
        return currentLoss

    


