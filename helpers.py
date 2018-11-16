# https://github.com/spro/char-rnn.pytorch

import string
import random
import time
import math
import torch


def read_file(filename):
    file = open(filename).read()
    all_characters = list(set(file))
    return file, len(file), all_characters, len(all_characters)

# Turning a string into a tensor

def char_tensor(string, all_characters):
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

