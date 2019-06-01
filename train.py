#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm
import string

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--dropout', type=float, default=0.3)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--batch_type', type=int, default=0)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--modelname', type=str, default=None)
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)
modelName = args.modelname

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)

        # while(file[start_index]!='\n'):  # first word should be the actual start of a sentence.
        #     start_index = start_index+1

        end_index = start_index + chunk_len + 1

        if(end_index>file_len): # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len-chunk_len-1

        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def consequent_training_set(chunk_len, batch_size, num_batches):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    end_index = chunk_len*num_batches*batch_size + (batch_size*num_batches)
    end_reached = False
    for bi in range(batch_size):
        start_index = end_index

        if (end_reached == True):
            start_index = random.randint(0, file_len - chunk_len - 1)

        if (start_index + chunk_len + 1 > file_len):  # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len - chunk_len - 1
            end_reached = True

        end_index = start_index + chunk_len + 1 # Adding 1 to create target
        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):

    decoder.zero_grad()
    loss = 0
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])

    ### The losses are averaged across observations for each minibatch (see doc CrossEntropyLoss)

    loss.backward()
    decoder_optimizer.step()
    currentLoss = loss.item()
    return currentLoss

def save():
    if modelName is not None:
        save_filename = os.path.splitext(os.path.basename(args.filename))[0] +modelName+ '.pt'
    else:
        save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)





# Initialize models and start training

all_characters = string.printable
n_characters = len(all_characters)

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
    dropout=args.dropout
)


decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
if args.cuda:
    decoder.cuda()

batch_type = args.batch_type

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    numFileBatches = math.ceil(file_len/((args.batch_size*args.chunk_len)+args.batch_size))
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        # end_index = 0
        numBatches = 0
        while(numBatches < numFileBatches) :
            if(batch_type == 0): ### Sampling batches at random
                loss = train(*random_training_set(args.chunk_len, args.batch_size))
            elif(batch_type == 1): ### Get consequent batches of chars without replacement
                loss = train(*consequent_training_set(args.chunk_len, args.batch_size, numBatches))
            loss_avg += loss
            numBatches += 1
        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Renzi', 200, cuda=args.cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

