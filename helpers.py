# https://github.com/spro/char-rnn.pytorch

import unidecode
import random
import time
import math
import torch

# Reading and un-unicode-encoding data

def read_file(filename):
    file = unidecode.unidecode(open(filename, encoding="utf8").read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

