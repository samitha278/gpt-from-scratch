import torch
import torch.nn as nn
import torch.nn.functional as F



#hyperparameters
block_size = 8
batch_size = 32
n_embd = 64
head_size = 16
eval_iter = 10000
max_iter = 10000
lr = 1e-3




with open('data.input.txt', 'r') as f:
    text = f.read()
    
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])











