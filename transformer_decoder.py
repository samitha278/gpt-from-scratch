import torch
import torch.nn as nn
import torch.nn.functional as F 


# hyperparameters
batch_size = 32
block_size = 64
eval_iters = 10000
n_embd = 128



with open('data/imput.txt', 'r') as f:
    text = f.read()
    
      
chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]



def get_batch(split):
    
    data = train if split=='train' else val
    
    ix = torch.randint(len(data)-block_size,(batch_size,))
    
    x = torch.tensor([data[i:i+block_size] for i in ix]) 
    y = torch.tensor([data[i+1:i+1+block_size] for i in ix])
    
    return x,y







# ----------------------------------------------------------

class transformerDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embd_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embd_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
        
        
    def forward(self,input, targets = None):
        
        B,T = input.shape
        
        token_embd = self.embd_table(input)
        pos_embd = self.pos_embd_table(torch.arange(T))
        
        x = token_embd+pos_embd
        
        logits = self.lm_head(x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ----------------------------------------------------------


model = transformerDecoder()




# model evaluation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
    
    







