import torch
import torch.nn as nn
import torch.nn.functional as F



#hyperparameters
block_size = 8
batch_size = 32
n_embd = 64
num_heads = 4
head_size = n_embd // num_heads
eval_iter = 10000
max_iter = 10000
lr = 1e-3



#text read
with open('data/input.txt', 'r') as f:
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
        out = torch.cat([sa(x) for sa in self.sa_heads],dim = -1)
        
        out = self.projection(out)
        
        return out
    
    
    

class MLP(nn.Module):
    
    def __init__(self,n_embd):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU()
        )

    def forward(self,x):
        
        return self.mlp(x)
        
    
     
    
class Block(nn.Module):
    
    def __init__(self,num_heads,n_embd):
        super().__init__()
        
        head_size = n_embd // num_heads
        
        self.heads = MultiHead(num_heads,head_size)
        self.mlp = MLP(n_embd)
        
    
    def forward(self,x):
        out = self.heads(x) + x  # residual conn
        out = self.mlp(out) + out  # residual conn
        
        return out 
        
           
        
        
        
    
    
class GPTModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embd_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embd_table = nn.Embedding(block_size,n_embd)
        
        self.block = nn.Sequential(
            Block(num_heads,n_embd),
            Block(num_heads,n_embd),
            Block(num_heads,n_embd)
        )
        
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
        
        
        
    def forward(self,input, targets = None):
        
        B,T = input.shape
        
        token_embd = self.embd_table(input)
        pos_embd = self.pos_embd_table(torch.arange(T))
        
        x = token_embd+pos_embd
        
        x = self.block(x)
        
        logits = self.lm_head(x)
        
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C),targets.view(-1))
            
        return logits , loss
        
        
        
         
        
    def train(self):
        
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for i in range(max_iter):
            
            xb,yb = get_batch('train')
            
            # evaluate model
            logits , loss = self(xb,yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            
            if i% (max_iter/10) == 0:
                print(f'{i}/{max_iter}  {loss}')
            if i == max_iter-1:
                print(f'{max_iter}/{max_iter}  {loss}')
            
        
                
                
                
    
    def generate(self,input,max_token):
        
        for _ in range(max_token):
            input_cond = input[:,-block_size:]
            logits , loss = self(input_cond)
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs,1)
            
            input = torch.cat((input,next_index),dim = 1)
            
        return input
        
        
       
# ----------------------------------------------------------



model = GPTModel()
model.train()



# genarate from the model 
context = torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(context,max_token=100)[0].tolist()))