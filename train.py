#!/usr/bin/env python
# https://github.com/zutotonno/char-rnn.pytorch

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

def random_dataset(args,file,file_len):
    inp = torch.LongTensor(args.batch_size, args.chunk_len)
    target = torch.LongTensor(args.batch_size, args.chunk_len)
    for bi in range(args.batch_size):
        start_index = random.randint(0, file_len - args.chunk_len)

        # while(file[start_index]!='\n'):  # first word should be the actual start of a sentence.
        #     start_index = start_index+1

        end_index = start_index + args.chunk_len + 1

        if(end_index>file_len): # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len-args.chunk_len-1

        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def consequent_dataset(args, num_batches, file, file_len):
    inp = torch.LongTensor(args.batch_size, args.chunk_len)
    target = torch.LongTensor(args.batch_size, args.chunk_len)
    end_index = args.chunk_len*num_batches*args.batch_size + (args.batch_size*num_batches)
    end_reached = False
    for bi in range(args.batch_size):
        start_index = end_index

        if (end_reached == True):
            start_index = random.randint(0, file_len - args.chunk_len - 1)

        if (start_index + args.chunk_len + 1 > file_len):  # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len - args.chunk_len - 1
            end_reached = True

        end_index = start_index + args.chunk_len + 1 # Adding 1 to create target
        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def save(args):
    save_filename = 'Save/'
    if modelName is not None:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '.pt'
    else:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] + '.pt'

    jsonName = save_filename + '.json'
    with open(jsonName, 'w') as json_file:
        json.dump(vars(args), json_file)
    saveLossesName = save_filename+'.csv'
    if(args.valid is not None):
        np.savetxt(saveLossesName, np.column_stack((train_losses, valid_losses)), delimiter=",", fmt='%s', header='Train,Valid')
    else:
        np.savetxt(saveLossesName, train_losses, delimiter=",", fmt='%s', header='Train')
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def savemodel(args):
    save_filename = 'Save/'
    directoryCheckpoint = 'Save/'+modelName
    if not os.path.exists(directoryCheckpoint):
        os.makedirs(directoryCheckpoint)
    if modelName is not None:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '_Checkpoint' +'.pt'
    else:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] + '_Checkpoint'+'.pt'

    torch.save(decoder, directoryCheckpoint)




# Initialize models and start training

if __name__ == '__main__':

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
    argparser.add_argument('--early_stopping', type=int, default=10)
    argparser.add_argument('--optimizer', type=str, default="adam")
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--modelname', type=str, default=None)
    args = argparser.parse_args()
    if args.cuda:
        print("Using CUDA")

    fileTrain, file_lenTrain = read_file(args.train)
    try:
        fileValid, file_lenValid = read_file(args.valid)
        early_stopping_patience = args.early_stopping
    except:
        print('No validation data supplied')
    if(args.modelname is None):
        print('No model name supplied -> Model checkpoint disabled')
    modelName = args.modelname

    all_characters = string.printable
    n_characters = len(all_characters)

    decoder = CharRNN(
        n_characters,
        args.hidden_size,
        n_characters,
        model=args.model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        chunk_len= args.chunk_len,
        batch_size=args.batch_size,
        gpu = args.cuda,
        optimizer = args.optimizer
    )




    batch_type = args.batch_type

    start = time.time()
    train_losses = []
    valid_losses = []
    valid_loss_best = np.inf
    patience = 1
    try:
        print("Training for %d epochs..." % args.n_epochs)
        numFileBatches = math.ceil(file_lenTrain/((args.batch_size*args.chunk_len)+args.batch_size))
        numValidBatches = math.ceil(file_lenValid/((args.batch_size*args.chunk_len)+args.batch_size))

        for epoch in tqdm(range(1, args.n_epochs + 1)):
            # end_index = 0
            numBatches = 0
            numBatchesValid = 0
            loss_avg = 0
            while(numBatches < numFileBatches) :
                if(batch_type == 0): ### Sampling batches at random
                    loss = decoder.train(*random_dataset(args,fileTrain,file_lenTrain),validation=False)
                elif(batch_type == 1): ### Get consequent batches of chars without replacement
                    loss = decoder.train(*consequent_dataset(args, numBatches,fileTrain, file_lenTrain),validation=False)
                loss_avg += loss
                numBatches += 1
            loss_avg /= numFileBatches
            train_losses.append(loss_avg)
            if args.valid is not None:
                valid_loss_avg = 0
                while(numBatchesValid < numValidBatches) :
                    valid_loss_avg += decoder.train(*consequent_dataset(args,numBatchesValid,fileValid,file_lenValid),validation=True)
                    numBatchesValid += 1
                valid_loss_avg /= numValidBatches
                valid_losses.append(valid_loss_avg)
                if valid_loss_avg < valid_loss_best:
                    if(args.modelname is not None):
                        print("New best checkpoint: %.4f, old: %.4f" % (valid_loss_avg,valid_loss_best))
                        savemodel(args)
                    valid_loss_best = valid_loss_avg
                    args.early_stopping = epoch
                    patience = 1
                else:
                    patience += 1
                    if(patience >= early_stopping_patience):
                        break

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) Train: %.4f Valid: %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_avg, valid_loss_avg))
                print(generate(decoder, 'Renzi', 200, cuda=args.cuda), '\n')

        print("Saving...")
        save(args)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(args)

