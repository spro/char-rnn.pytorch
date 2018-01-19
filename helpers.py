# https://github.com/spro/char-rnn.pytorch

import string
import random
import time
import math
import torch

# Reading and extracting vocab from data

def read_file(filename):
    file = open(filename).read()
    vocab = ''.join(sorted(set(file)))
    return file, len(file), vocab

# Turning a string into a tensor

def char_tensor(string, vocab):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = vocab.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

