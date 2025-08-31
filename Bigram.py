import torch
import torch.nn.functional as F





with open("data/input.txt",'r',encoding='utf-8') as f:
    text = f.read()
    
# print(text[:10])



# character mapping 
chars = sorted(list(set(text)))
vocab_size = len(chars)



stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda w: ''.join([itos[i] for i in w])



# text mapping




# split dataset train/val



# get batch



