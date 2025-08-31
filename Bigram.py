import torch
import torch.nn.functional as F



# hyperpapameters 
batch_size = 32
block_size = 8



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
data = torch.tensor(encode(text) , dtype=torch.long)


# split dataset train/val
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]


# minibatch
def get_batch(split):
    
    data = train if split == "train" else val
    
    n = len(data)
    
    ix = torch.randint(n - block_size , (batch_size,))
    
    x = torch.tensor(data[i:i+block_size] for i in ix)
    y = torch.tensor(data[i+1:i+block_size+1] for i in ix)
    
    
    return x,y



    
    












# get batch



