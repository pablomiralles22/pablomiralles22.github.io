---
layout: post
title: Coding a fused matrix multiplication and log-sum-exp reduction in Triton to retrieve LLM log-probabilities faster and with lower memory overhead
description: Fusing matmul + log-sum-exp in Triton to read LLM token log-probabilities efficiently (less memory, faster)
date: 2025-12-11

tags:
- deep-learning
- triton
---

### Motivation
In my research I have worked with methods that involve retrieving the log-probabilities of an input text from a large language model. Let me first introduce what I mean exactly and why this is useful.

Consider an input text tokenized as a sequence $t_1,\dots,t_L$. Each token is mapped to an initial embedding, forming the initial hidden-state matrix $\mathrm{HS}^0 \in \mathbb{R}^{L \times D}$. The transformer then processes these representations and produces the contextualized final hidden states

$$
f(\mathrm{HS}^0;\theta) = \mathrm{HS}^{\text{final}} \in \mathbb{R}^{L \times D}.
$$

A final linear layer with weights $W \in \mathbb{R}^{V \times D}$ and bias $b \in \mathbb{R}^{V}$ is applied to obtain the logits

$$
\mathrm{logits} = \mathrm{HS}^{\text{final}}\cdot W^T + b \in \mathbb{R}^{L \times V}.
$$

Applying the softmax along the vocabulary dimension yields the probability matrix

$$
P = \mathrm{softmax}(\mathrm{logits},\ \mathrm{dim}=1) \in \mathbb{R}^{L \times V}.
$$

This probability matrix encodes the model’s next-token predictions. For each position $i \in {1,\dots,L}$, the vector $P_i \in \mathbb{R}^{V}$ is a valid probability distribution (non-negative and summing to one) over the vocabulary. Its $v$-th component represents the model’s estimated probability that the next token is the $v$-th vocabulary item, conditioned on the preceding context $t_1,\dots,t_i$.

During the generation process we only compute the final probability vector to sample the new token to be appended. However, we can apply the model to a full input text and study these probabilities across tokens. This is useful in distinguishing AI-generated text (we expect these texts to score higher when passed through the LLM), and we have also leveraged them for human authorship problems.

### Computational problems
The vocabulary of modern LLMs is very large (100K–300K tokens), and the matrix $P \in \mathbb{R}^{L \times V}$ becomes extremely large for long texts, especially in batched settings. Often we only need the log-probability of the token that actually occurred next in the sequence; i.e., we want the sequence 

$$
\log P_{1, t_2},\ \log P_{2, t_3},\ \dots,\ \log P_{L-1, t_L}.
$$

If we compute these values smartly, we avoid instantiating the full probability or logit matrix in memory!

### Derivations and algorithm

Now, we can consider the following equivalence:

$$
\log P
= \mathrm{logsoftmax}(\mathrm{logits},\ \mathrm{dim}=1)
= \mathrm{logits} - \mathrm{logsumexp}(\mathrm{logits},\ \mathrm{dim}=1),
$$

and therefore

$$
\log P_{i, t_{i+1}}
= \mathrm{logits}_{i, t_{i+1}} - \mathrm{logsumexp}(\mathrm{logits}_i),
$$

The first component is actually very easy to simplify:

$$
\mathrm{logits}_{i, t_{i+1}} =
\mathrm{HS}^{\text{final}}_i \cdot W_{t_{i+1}} + b_{t_{i+1}}.
$$

The second term is a reduction across the vocabulary dimension. In other words, $\mathrm{logsumexp}(\mathrm{logits},\ \mathrm{dim}=1)$ yields a length-$L$ vector, not a massive matrix. Naively, you’d instantiate the full logits to compute `logsumexp`, which is expensive. Instead, we can fuse the linear layer that produces logits with the `logsumexp` reduction in a single kernel.

### Tiled matmul & online log-sum-exp
Consider an input matrix $A\in \mathbb R^{M \times K}$, a weight matrix $B\in \mathbb R^{N \times K}$ (we ignore the bias vector for simplicity here). We want to implement the following:

$$
C = \mathrm{logsumexp} (A \cdot B^T, \mathrm{dim} = 1),
$$

or, in non-vectorial form:

$$
C_i = \log \sum_{j=1}^N \exp(A_i \cdot B_j^T); \qquad i=1,\dots M.
$$

As with general matrix multiplication, we tile the matrices; each tile pair is handled by a tensor core. While computing tile products, we apply the exponential and maintain the running sum of exponentials across the $N$ dimension. After accumulation along $N$, we take the logarithm.

```python
for m in range(0, M, BLOCK_SIZE_M):
    reduce_global = zeros((BLOCK_SIZE_M,), dtype=float32)
    
    for n in range(0, N, BLOCK_SIZE_N):
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[n : n+BLOCK_SIZE_N, k : k+BLOCK_SIZE_K]
            acc += dot(a, b.T)

        reduce_global += sum(exp(acc), axis=1)
    
    C = log(reduce_global)
```

After dividing the algorithm in blocks, we can parallelize the program across tensor cores by blocks. Since we have a reduction dependency in both the $N$ and $K$ dimensions, we will parallelize across blocks in the $M$ dimension. However, there is an additional implementation detail that is typically implemented for numerical stability, which is mathematically equivalent but avoids computing the exponential of large numbers:

$$
\begin{align}
M &= \max (A \cdot B^T,\ \mathrm{dim}=1) \\
C &= M + \mathrm{logsumexp} (A \cdot B^T - M,\ \mathrm{dim} = 1).
\end{align}
$$

This slightly complicates computation across $N$-tiles: we must track the online maximum and the sum of exponentials with the maximum subtracted. The algorithm:

```python
for m in range(0, M, BLOCK_SIZE_M):
    reduce_global = zeros((BLOCK_SIZE_M,), dtype=float32)
    max_global = full((BLOCK_SIZE_M,), float('-inf'), dtype=float32)
    
    for n in range(0, N, BLOCK_SIZE_N):
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[n : n+BLOCK_SIZE_N, k : k+BLOCK_SIZE_K]
            acc += dot(a, b.T)

        max_block = max(acc, axis=1)
        acc = exp(acc - max_block[:, None])
        reduce_block = sum(acc, axis=1)

        # Update global values
        max_global_new = maximum(max_global, max_block)
        reduce_global = (
            reduce_global * exp(max_global - max_global_new) +
            reduce_block * exp(max_block - max_global_new)
        )
        max_global = max_global_new
    
    C = max_global + log(reduce_global)
```

The full code is available at [pablomiralles22/fused-matmul-logsumexp](https://github.com/pablomiralles22/fused-matmul-logsumexp), and you will find a good description and instructions on how to use it on the `README` file.

### Benchmarking
Consider an input matrix $X\in \mathbb R^{M \times K}$, a weight matrix $W\in \mathbb R^{N \times K}$, and a bias vector $b\in \mathbb R^{N}$. The following plots show execution time and peak memory for varying $M$ and $N$, comparing the fused implementation to a naive baseline:

{% include figure.liquid path="assets/img/posts/2025-12-11-fused-matmul-logsumexp/benchmark_vs_N.png" class="img-fluid rounded z-depth-1" %}
{% include figure.liquid path="assets/img/posts/2025-12-11-fused-matmul-logsumexp/benchmark_vs_M.png" class="img-fluid rounded z-depth-1" %}