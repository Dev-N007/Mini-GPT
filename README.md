# MiniGPT : Reduced Decoder-Only Transformer (GPT-2) from scratch

## Overview

This project implements a reduced version of the GPT-2 decoder-only Transformer architecture from first principles using PyTorch. The objective was to deeply understand how autoregressive language models are built internally instead of relying on high-level transformer abstractions.

The GPT-2 tokenizer from HuggingFace is used for vocabulary compatibility, while the entire transformer architecture and training pipeline are implemented manually.

The implementation includes:

- Masked multi-head self-attention  
- Feed-forward networks with GELU activation  
- Residual connections  
- Pre-LayerNorm transformer blocks  
- Weight tying  
- Autoregressive next-token prediction  

---

## Motivation

Large language models such as GPT-2 demonstrate how decoder-only transformers model long-range dependencies in natural language.

The goals of this project were:

- To understand masked self-attention mathematically and programmatically  
- To implement GPT-2 architecture from scratch  
- To build a complete autoregressive training pipeline  
- To evaluate model performance using perplexity  
- To validate structural correctness against the official GPT-2 implementation  

---

## Model Architecture

This implementation follows the GPT-2 architecture described in *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019), with reduced dimensions for practical training.

### Configuration (Reduced Version)

- 6 Transformer blocks  
- Embedding dimension: 512  
- 8 attention heads  
- Context window: 512 tokens  
- Vocabulary size: 50257  

The structural design remains identical to GPT-2 Small, but parameters are reduced to allow training on consumer hardware.

---

## Architecture Breakdown

### 1. Tokenization

GPT-2 uses byte-level Byte Pair Encoding (BPE).  
Instead of reimplementing BPE, the official GPT-2 tokenizer was used to ensure:

- Correct vocabulary size  
- Proper byte-level encoding  
- Compatibility with pretrained GPT-2  

---

### 2. Input Representation

Each input sequence is transformed as:

Input IDs  
→ Token Embedding (wte)  
→ Positional Embedding (wpe)  
→ Dropout  

Learned positional embeddings are used instead of sinusoidal encoding.

Output shape:

(batch_size, sequence_length, embedding_dimension)

---

### 3. Masked Multi-Head Self-Attention

The attention module implements:

- Combined QKV projection  
- Multi-head splitting  
- Scaled dot-product attention  
- Causal masking  
- Output projection  

Mathematically:

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V  

Causal masking ensures token *t* cannot attend to tokens greater than *t*, preserving autoregressive behavior.

---

### 4. Feed-Forward Network (MLP)

Each transformer block includes a position-wise feed-forward network:

- Linear (n_embd → 4 × n_embd)  
- GELU activation  
- Linear (4 × n_embd → n_embd)  

The 4× expansion factor follows the original GPT-2 architecture.

---

### 5. Residual Connections and Pre-LayerNorm

Each block follows the Pre-LayerNorm structure:
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Pre-LayerNorm improves training stability in deeper transformers.

---

### 6. Weight Tying

The output projection layer shares weights with the token embedding matrix:

lm_head.weight = embedding.weight


This reduces parameter count and improves generalization.

---

### 7. Initialization

All linear and embedding layers are initialized using:

    Normal(0, 0.02)

This is as per GPT-2 specifications.

---

## Training Setup

### Objective

Autoregressive next-token prediction.

For a token sequence:

[x₁, x₂, x₃, x₄]

The model learns:

P(x₂ | x₁)  
P(x₃ | x₁, x₂)  
P(x₄ | x₁, x₂, x₃)

Loss function: CrossEntropyLoss over vocabulary logits.

---

### Optimizer

- AdamW  
- Learning rate: 3e-4  
- Gradient clipping applied for stability  

Perplexity is computed as:
    perplexity = exp(loss)

---

## Dataset

The model was trained on WikiText (raw version), converted into a continuous text stream and tokenized using the GPT-2 tokenizer.

Block-based slicing was used to create autoregressive input-target pairs.

---

## Text Generation

After training, text generation proceeds autoregressively:

1. Feed prompt tokens  
2. Predict next-token distribution  
3. Sample from softmax probabilities  
4. Append token  
5. Repeat  

Multinomial sampling is used for stochastic generation.

---

## What This Project Demonstrates

- Deep understanding of transformer internals  
- Manual implementation of masked attention  
- Knowledge of autoregressive language modeling  
- End-to-end training and evaluation pipeline  
- Practical debugging of ML systems  

---

## Limitations

- Reduced model size (not full GPT-2 124M parameters)  
- Trained on smaller corpus compared to original GPT-2  
- No distributed training  

---

## Future Improvements

- Learning rate scheduler (cosine decay)  
- Mixed precision training  
- Top-k and nucleus sampling  
- FlashAttention  
- LoRA fine-tuning  
- Larger-scale training  

---

## Tools and Technologies

- Python  
- PyTorch  
- HuggingFace Transformers (Tokenizer only)  
- HuggingFace Datasets  
- NumPy  
- Pandas  
- Matplotlib  
- Jupyter Notebook  
