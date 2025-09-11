# GPT from Scratch

A PyTorch implementation of a Generatively Pretrained Transformer (GPT) built from the ground up, inspired by the "Attention is All You Need" paper and OpenAI's GPT-2/3 architecture.

## Overview

This project implements the complete pipeline for autoregressive language modeling, including:
- Model architecture implementation
- Training procedures
- Text generation/sampling
- Model evaluation

#### Self Attention (simple explanation)
```
Key = resume ("Here are my skills and qualities")
Query = job description ("I'm looking for someone with these skills")
dot(Query,Key) = how well the resume matches the job description
```

## Implementation Progress

### Step 1: Bigram Language Model
**File:** `Bigram.py`

A simple bigram model serving as the foundation before adding transformer components.

**Training Results:**
```
Step    Loss
0       4.546
1000    3.676
2000    3.054
3000    2.730
4000    2.556
5000    2.520
6000    2.589
7000    2.403
8000    2.335
9000    2.460
```

**Sample Generation (100 tokens):**
```
Fours thid J ous?          
Bouelllllighapan ITh.      
I s.                       
LI:                        
The;                       
                           
Fary be be bu uneDortanethethe      
Foan tha                            
nchee                               
```
*Note: Output quality is limited due to the simple bigram approach*

### Step 2: Single Self-Attention Head
**File:** `single_attention.py`

Added a single self-attention mechanism to improve context understanding.

**Training Results:**
```
Step    Loss
0       4.210
1000    2.466
2000    2.476
3000    2.479
4000    2.425
5000    2.337
6000    2.290
7000    2.260
8000    2.311
9000    2.239
```

**Sample Generation (100 tokens):**
```
HAMTER:                                 
K maz: 
Bot I'lte se we my, 'd die cenotirh,
I'n:
Met,
Shave malke sherd.
Pa'd Golle ave ye X
```
*Improved coherence with attention mechanism*



### Step 3: Multi-Head Attention Mechanism
**File:** `multi_head_attn.py`

Adding a Multi-Head Attention Mechanism to improve context understanding further.

**Training Results:**
```
0/10000  4.186285018920898
1000/10000  2.49324107170105
2000/10000  2.2129158973693848
3000/10000  2.2964208126068115
4000/10000  2.1379356384277344
5000/10000  1.9161547422409058
6000/10000  2.146939277648926
7000/10000  2.1689884662628174
8000/10000  2.5390045642852783
9000/10000  2.099254608154297
```

**Sample Generation (100 tokens):**
```
As youblead, my dimy obelcersing,
Hall in thous ti wing meged Cad ticeds yould with I thath dy me my
```

*Improved with multi head attention mechanism*


## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- OpenAI GPT-2/3 architectures

