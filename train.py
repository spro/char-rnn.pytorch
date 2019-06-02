#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import string

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--train', type=str)
argparser.add_argument('--valid', type=str)
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

fileTrain, file_lenTrain = read_file(args.train)
fileValid, file_lenValid = read_file(args.valid)
modelName = args.modelname


def random_dataset(chunk_len, batch_size,file,file_len):
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
            start_index = random.randint(0, file_lenTrain - chunk_len - 1)

        if (start_index + chunk_len + 1 > file_lenTrain):  # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_lenTrain - chunk_len - 1
            end_reached = True

        end_index = start_index + chunk_len + 1 # Adding 1 to create target
        chunk = fileTrain[start_index:end_index]

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
        if args.model == "gru":
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])

    ### The losses are averaged across observations for each minibatch (see doc CrossEntropyLoss)

    loss.backward()
    decoder_optimizer.step()
    currentLoss = loss.item()/ args.chunk_len
    return currentLoss

def valid(inp,target):
    decoder.zero_grad()
    loss = 0
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        if args.model == "gru":
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])
    currentLoss = loss.item() / args.chunk_len
    return currentLoss

def save():
    save_filename = 'Save/'
    if modelName is not None:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '.pt'
    else:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] + '.pt'

    jsonName = save_filename + '.json'
    with open(jsonName, 'w') as json_file:
        json.dump(vars(args), json_file)
    saveLossesName = save_filename+'.csv'
    np.savetxt(saveLossesName, np.column_stack((train_losses, valid_losses)), delimiter=",", fmt='%s', header='Train,Valid')
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def savemodel(epoch):
    save_filename = 'Save/'
    directoryCheckpoint = 'Save/'+modelName
    if not os.path.exists(directoryCheckpoint):
        os.makedirs(directoryCheckpoint)
    if modelName is not None:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '_'+str(epoch) +'.pt'
    else:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] + '_'+str(epoch)+'.pt'

    torch.save(decoder, directoryCheckpoint)




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
train_losses = []
valid_losses = []
loss_avg = 0
valid_loss_avg = 0
try:
    print("Training for %d epochs..." % args.n_epochs)
    numFileBatches = math.ceil(file_lenTrain/((args.batch_size*args.chunk_len)+args.batch_size))
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        # end_index = 0
        numBatches = 0

        while(numBatches < numFileBatches) :
            if(batch_type == 0): ### Sampling batches at random
                loss = train(*random_dataset(args.chunk_len, args.batch_size,fileTrain,file_lenTrain))
            elif(batch_type == 1): ### Get consequent batches of chars without replacement
                loss = train(*consequent_training_set(args.chunk_len, args.batch_size, numBatches))
            loss_avg += loss
            numBatches += 1
        loss_avg /= numFileBatches
        valid_loss_avg = valid(*random_dataset(args.chunk_len, args.batch_size,fileValid,file_lenValid))
        
        train_losses.append(loss_avg)
        valid_losses.append(valid_loss_avg)
        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) Train: %.4f Valid: %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_avg, valid_loss_avg))
            print(generate(decoder, 'Renzi', 200, cuda=args.cuda), '\n')
            savemodel(epoch)

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

