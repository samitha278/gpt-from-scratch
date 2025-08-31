import torch
import torch.nn as nn
import torch.nn.functional as F



# hyperpapameters 
batch_size = 32
block_size = 8
learning_rate = 0.001
max_iter = 10000



with open("data/input.txt",'r',encoding='utf-8') as f:
    text = f.read()
    

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
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y


#-----------------------------------------------------------------

# Bigram Language Model
class BigramLM(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size,vocab_size)
        
        
    def forward(self,index,target=None):
        
        logits = self.token_emb_table(index)
        
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(-1,C),target.view(-1))
            
        return logits , loss
    
    
    def generate(self,index,max_token):
        
        for _ in range(max_token):
            logits , loss = self(index)
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs,1)
            
            index = torch.cat((index,next_index),dim = 1)
            
        return index
    
#------------------------------------------------------------------  


model = BigramLM(vocab_size)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



for i in range(max_iter):
    
    xb,yb = get_batch('train')
    
    # evaluate model
    logits , loss = model(xb,yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
    if i% (max_iter/10) == 0:
        print(f'{i}/{max_iter}  {loss}')
    
    
    
# first run output
"""
0/10000  4.769690036773682
1000/10000  3.861417055130005
2000/10000  3.132174015045166
3000/10000  2.8828563690185547
4000/10000  2.784301519393921
5000/10000  2.471895217895508
6000/10000  2.762051820755005
7000/10000  2.4801313877105713
8000/10000  2.5372257232666016
9000/10000  2.5602262020111084
"""




   
