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



#text read
with open('data.input.txt', 'r') as f:
    text = f.read()
    

#get characters from input.txt
chars = sorted(list(set(text)))
vocab_size = len(chars)


#encoding and decoding
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])



# encode text
data = encode(text)

# train / val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



# create mini batch
def get_batch(split):
    
    data = train_data if split=='train' else val_data
    
    idx = torch.randint(len(data)-block_size , (block_size,))
    
    x = torch.tensor([data[i:i+block_size] for i in idx])
    y = torch.tensor([data[i+1:block_size+1+i] for i in idx])
    
    return x,y




# -------------------------------------------------------------
    

class SaHead(nn.Module):
    
    
    def __init__(self,head_size):
        super().__init__()

        self.key = nn.Linear(n_embd,head_size, bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        
        self.value = nn.Linear(n_embd,head_size, bias = False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        
    def forward(self,x):
        
        B,T,C = x.shape
        
        key = self.key(x)
        query = self.query(x)
        
        weight = query @ key.transpose(-2,-1) * C**-0.5
        weight = weight.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        weight = F.softmax(weight,dim=-1)
        
        value = self.value(x)
        
        out = weight @ value
        
        return out
        
        
class MultiHead(nn.Module):
    
    def __init__(self, num_heads,head_size):
        super().__init__()
        
        self.sa_heads = nn.ModuleList([SaHead(head_size) for i in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size , n_embd)
        
        
        
    def forward(self,x):
        self.out = torch.cat([sa(x) for sa in self.sa_heads],dim = -1)
        
        self.out = self.projection(self.out)
        
        return self.out