# Generatively Pretrained Transformer from scratch

Implementing a Generatively Pretrained Transformer (GPT) from scratch in PyTorch, inspired by Attention is All You Need paper and OpenAI’s GPT-2/3. Covers the full pipeline of autoregressive language modeling, training, sampling, and evaluation.







#### Step 1
##### Bigram Language Model (Bigram.py)
###### Outout:
0/10000  4.5462846755981445    <br/>
1000/10000  3.6757185459136963 <br/>
2000/10000  3.053956985473633  <br/>
3000/10000  2.7299282550811768  <br/>
4000/10000  2.556368827819824   <br/>
5000/10000  2.520214796066284   <br/>
6000/10000  2.5889151096343994 <br/>
7000/10000  2.4034910202026367 <br/>
8000/10000  2.3354899883270264 <br/>
9000/10000  2.4603657722473145 <br/>


###### Generate 100 tokens: (kinda garbage, need to improve this model)
Fours thid J ous?          <br/> 
Bouelllllighapan ITh.      <br/>
I s.                       <br/>
LI:                        <br/>
The;                       <br/>
                           <br/>
Fary be be bu uneDortanethethe      <br/>
Foan tha                            <br/>
nchee                               <br/>



#### Step 2
##### Add one self attention head (transformer_decoder.py)
###### Outout:
0/10000  4.152102470397949       <br/>
1000/10000  2.5086829662323      <br/>
2000/10000  2.4729814529418945  <br/>
3000/10000  2.542850971221924  <br/>
4000/10000  2.482938289642334  <br/>
5000/10000  2.5032737255096436  <br/>
6000/10000  2.4475369453430176  <br/>
7000/10000  2.462331771850586  <br/>
8000/10000  2.487802505493164  <br/>
9000/10000  2.4615895748138428  <br/>

###### Generate 100 tokens:
F y w oteaco hot'thay it imeal-damig her: ndaus or sll, swin odr ut:     <br/>
Moghend me fr tousee thesend y-                                          <br/>



## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


