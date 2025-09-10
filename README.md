# GPT from Scratch

A PyTorch implementation of a Generatively Pretrained Transformer (GPT) built from the ground up, inspired by the "Attention is All You Need" paper and OpenAI's GPT-2/3 architecture.

## Overview

This project implements the complete pipeline for autoregressive language modeling, including:
- Model architecture implementation
- Training procedures
- Text generation/sampling
- Model evaluation

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

Adding a Multi-Head Attention Mechanism to improve context understanding further. ongoing...


## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- OpenAI GPT-2/3 architectures

