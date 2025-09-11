import torch
import torch.nn as nn
import torch.nn.functional as F 


# hyperparameters
batch_size = 32
block_size = 64
eval_iters = 10000
n_embd = 128
head_size = 16
max_iter  = 10000
learning_rate = 1e-2



with open('data/input.txt', 'r') as f:
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




# minibatch
def get_batch(split):
    
    data = train if split == "train" else val
    n = len(data)
    ix = torch.randint(n - block_size , (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y







# ----------------------------------------------------------
class selfAttentionHead(nn.Module):
    
    
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
        
        
        
        
        

class transformerDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embd_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embd_table = nn.Embedding(block_size,n_embd)
        
        self.sa_head = selfAttentionHead(head_size)
        
        self.lm_head = nn.Linear(head_size,vocab_size)
        
        
        
        
    def forward(self,input, targets = None):
        
        B,T = input.shape
        
        token_embd = self.embd_table(input)
        pos_embd = self.pos_embd_table(torch.arange(T))
        
        x = token_embd+pos_embd
        x = self.sa_head(x)
        
        logits = self.lm_head(x)
        
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C),targets.view(-1))
            
        return logits , loss
        
        
        
        
        
    def generate(self,input,max_token):
        
        for _ in range(max_token):
            input_cond = input[:,-block_size:]
            logits , loss = self(input_cond)
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs,1)
            
            input = torch.cat((input,next_index),dim = 1)
            
        return input
        
        
    def train(self):
        
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for i in range(max_iter):
            
            xb,yb = get_batch('train')
            
            # evaluate model
            logits , loss = model(xb,yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            
            if i% (max_iter/10) == 0:
                print(f'{i}/{max_iter}  {loss}')
        
        
       
# ----------------------------------------------------------





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
    
    
    
    
    
    
    
    
# Train model    
model = transformerDecoder()
model.train()


    


# genarate from the model 
context = torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(context,max_token=100)[0].tolist()))



