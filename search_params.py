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
import itertools

from helpers import *
from model import *
from generate import *

def random_dataset(program_args,args,file,file_len):
    inp = torch.LongTensor(args['batch_size'], args['chunk_len'])
    target = torch.LongTensor(args['batch_size'], args['chunk_len'])
    for bi in range(args['batch_size']):
        start_index = random.randint(0, file_len -args['chunk_len'])

        # while(file[start_index]!='\n'):  # first word should be the actual start of a sentence.
        #     start_index = start_index+1

        end_index = start_index + args['chunk_len'] + 1

        if(end_index>file_len): # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len-args['chunk_len']-1

        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if program_args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def consequent_dataset(program_args,args, num_batches, file, file_len):
    inp = torch.LongTensor(args['batch_size'], args['chunk_len'])
    target = torch.LongTensor(args['batch_size'], args['chunk_len'])
    end_index = args['chunk_len']*num_batches*args['batch_size'] + (args['batch_size']*num_batches)
    end_reached = False
    for bi in range(args['batch_size']):
        start_index = end_index

        if (end_reached == True):
            start_index = random.randint(0, file_len - args['chunk_len'] - 1)

        if (start_index + args['chunk_len'] + 1 > file_len):  # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len - args['chunk_len'] - 1
            end_reached = True

        end_index = start_index + args['chunk_len'] + 1 # Adding 1 to create target
        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if program_args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def savemodel(modelName,args):
    save_filename = 'Save/'
    directoryCheckpoint = 'Save/'+modelName
    if not os.path.exists(directoryCheckpoint):
        os.makedirs(directoryCheckpoint)
    if modelName is not None:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '_Checkpoint' +'.pt'
    else:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] + '_Checkpoint'+'.pt'

    torch.save(decoder, directoryCheckpoint)


def save(modelName,params,train_losses,valid_losses):
    save_filename = 'Save/'
    save_filename += modelName

    jsonName = save_filename + '.json'
    with open(jsonName, 'w') as json_file:
        json.dump(params, json_file)
    saveLossesName = save_filename+'.csv'
    if(valid_losses is not None):
        np.savetxt(saveLossesName, np.column_stack((train_losses, valid_losses)), delimiter=",", fmt='%s', header='Train,Valid')
    else:
        np.savetxt(saveLossesName, train_losses, delimiter=",", fmt='%s', header='Train')
    print('Saved as %s' % save_filename)


# Initialize models and start training

if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str)
    argparser.add_argument('--valid', type=str)

    argparser.add_argument('--hidden_size_init', type=int, default=50)
    argparser.add_argument('--hidden_size_end', type=int, default=800)
    argparser.add_argument('--hidden_size_step', type=int, default=200)
    
    argparser.add_argument('--n_layers_init', type=int, default=1)
    argparser.add_argument('--n_layers_end', type=int, default=4)
    argparser.add_argument('--n_layers_step', type=int, default=1)


    argparser.add_argument('--chunk_len_init', type=int, default=20)
    argparser.add_argument('--chunk_len_end', type=int, default=90)
    argparser.add_argument('--chunk_len_step', type=int, default=10)

    argparser.add_argument('--early_stopping', type=int, default=10)

    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--optimizer', type=str, default="adam")
    argparser.add_argument('--print_every', type=int, default=10)
    args = argparser.parse_args()

    if args.cuda:
        print("Using CUDA")

    fileTrain, file_lenTrain = read_file(args.train)
    try:
        fileValid, file_lenValid = read_file(args.valid)
        early_stopping_patience = args.early_stopping
    except:
        print('No validation data supplied')

    all_characters = string.printable
    n_characters = len(all_characters)

    params_list = []

    ##0
    n_epochs_list = [500]
    params_list.append(n_epochs_list)
    ##1
    n_hidden_list = list(range(args.hidden_size_init,args.hidden_size_end,args.hidden_size_step))
    params_list.append(n_hidden_list)
    ##2
    n_layers_list = list(range(args.n_layers_init,args.n_layers_end,args.n_layers_step))
    params_list.append(n_layers_list)

    # n_dropout_list = [0,0.3]
    # params_list.append(n_dropout_list)

    ##3
    n_chunk_len_list = list(range(args.chunk_len_init,args.chunk_len_end,args.chunk_len_step))
    params_list.append(n_chunk_len_list)
    ##4
    n_batch_size_list = [1024,2048]
    params_list.append(n_batch_size_list)
    ##5
    n_learning_rate_list = [0.001,0.01]
    params_list.append(n_learning_rate_list)
    ##6
    batch_type = [0]
    params_list.append(batch_type)
    ##7
    model_type = ['lstm']
    params_list.append(model_type)

    param_combinations = list(itertools.product(*params_list))

    currentCombination = 1
    for params in param_combinations :
        param_dict = dict()
        param_dict['model'] = params[-1]
        param_dict['hidden_size'] = params[1]
        param_dict['n_layers'] = params[2]
        param_dict['learning_rate'] = params[5]
        param_dict['chunk_len'] = params[3]
        param_dict['batch_size'] = params[4]
    
        decoder = CharRNN(
            input_size = n_characters,
            output_size = n_characters,
            **param_dict
        )


        param_dict['batch_type'] = params[6]
        param_dict['epochs'] = params[0]
        train_losses = []
        valid_losses = []
        loss_avg = 0
        valid_loss_avg = 0
        start = time.time()
        valid_loss_best = np.inf
        patience = 1

        try:
            print("Training for %d epochs..." % param_dict['epochs'])
            modelName = str(currentCombination)
            print(param_dict)
            numFileBatches = math.ceil(file_lenTrain/((param_dict['batch_size']*param_dict['chunk_len'])+param_dict['batch_size']))
            numValidBatches = math.ceil(file_lenValid/((param_dict['batch_size']*param_dict['chunk_len'])+param_dict['batch_size']))
            for epoch in tqdm(range(1, param_dict['epochs'] + 1)):
                # end_index = 0
                numBatches = 0
                numBatchesValid = 0
                loss_avg = 0
                while(numBatches < numFileBatches) :
                    if(param_dict['batch_type'] == 0): ### Sampling batches at random
                        loss = decoder.train(*random_dataset(args,param_dict,fileTrain,file_lenTrain),validation=False)
                    elif(param_dict['batch_type'] == 1): ### Get consequent batches of chars without replacement
                        loss = decoder.train(*consequent_dataset(args, param_dict, numBatches,fileTrain, file_lenTrain),validation=False)
                    loss_avg += loss
                    numBatches += 1
                loss_avg /= numFileBatches
                train_losses.append(loss_avg)
                if args.valid is not None:
                    valid_loss_avg = 0
                    while(numBatchesValid < numValidBatches) :
                        valid_loss_avg += decoder.train(*consequent_dataset(args,param_dict,numBatchesValid,fileValid,file_lenValid),validation=True)
                        numBatchesValid += 1
                    valid_loss_avg /= numValidBatches
                    valid_losses.append(valid_loss_avg)
                    if valid_loss_avg < valid_loss_best:
                        print("New best checkpoint: %.4f, old: %.4f" % (valid_loss_avg,valid_loss_best))
                        savemodel(modelName, args)
                        valid_loss_best = valid_loss_avg
                        args.early_stopping = epoch
                        patience = 1
                    else:
                        patience += 1
                        if(patience >= early_stopping_patience):
                            break
                if epoch % args.print_every == 0:
                    print('[%s (%d %d%%) Train: %.4f Valid: %.4f]' % (time_since(start), epoch, epoch / param_dict['epochs'] * 100, loss_avg, valid_loss_avg))
                    print(generate(decoder, 'Renzi', 200, cuda=args.cuda), '\n')

            print("Saving...")
            param_dict['early_stopping'] = args.early_stopping
            save(modelName,param_dict,train_losses,valid_losses)
            currentCombination += 1
        except KeyboardInterrupt:
            print("Saving before quit...")
            save(param_dict)

