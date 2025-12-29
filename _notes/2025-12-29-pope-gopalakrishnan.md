---
layout: post
title: Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings - Gopalakrishnan et al.
description: |
    This paper summary/commentary explores PoPE (Polar Coordinate Positional
    Embeddings), a novel approach to disentangling content and position in
    Transformers. By mapping key and query vectors into a polar space where
    content determines magnitude and position determines phase, PoPE addresses a
    fundamental limitation of Rotary Positional Embeddings (RoPE): the
    entanglement of semantic and spatial information.
date: 2025-12-29
author: Pablo Miralles
---

Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). _Decoupling the “What” and “Where” With Polar Coordinate Positional Embeddings_ (No. arXiv:2509.10534). arXiv. [https://doi.org/10.48550/arXiv.2509.10534](https://doi.org/10.48550/arXiv.2509.10534)

---
The paper proposes Polar Coordinate Position Embedding (PoPE), an attempt at improving the RoPE positional embeddings that are most commonly used in LLMs at the time of writing. I assume here that you are familiar with the latter. Otherwise, there are many amazing blog posts and youtube videos that explain them!

**Motivation.** Rotary Positional Embeddings (RoPE) rotates the 2D subcomponents of the keys and queries, entangling the angle from positional information with the angle from the content itself. They are trying to disentangle content from position.

**Method.**  PoPE can be seen as an application of RoPE with a previous transformation of the key and query vectors using the following function:

$$
\begin{array}{rcl}
f: & \mathbb{R}^d & \longrightarrow & \mathbb{R}^{2d} \\[10pt]
& \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} 
& \longmapsto 
& \begin{bmatrix} \mathrm{softplus}(x_1) \\ 0 \\ \mathrm{softplus}(x_2) \\ 0 \\ \vdots \\ \mathrm{softplus}(x_d) \\ 0 \end{bmatrix}
\end{array}.
$$

Thus, the 2D subcomponents of the vectors always have a positive first coordinate and a zero second coordinate, that is, the angle of all 2D subcomponents is always $0$. This means that only the position-based rotation affects the angle, and not the content.

The following is a pseudocode implementation of PoPE (single head for simplicity):
```python
# k: [B, L, D], q: [B, L, D], v: [B, L, D], theta: [D]
idxs = arange(0, L).view(1, L, 1)
angles = idxs * theta.view(1, 1, D)  # [1, L, D]

hat_k, hat_q = softplus(k), softplus(q)
hat_k = interleave(hat_k * cos(angles), hat_k * sin(angles))
hat_q = interleave(hat_q * cos(angles), hat_q * sin(angles))

return attention(hat_q, hat_k, v)
```

**Results.**
- Able to learn indirect indexing: given an input text, a source symbol in the text, and an offset, the model has to predict the symbol at the offset position from the source symbol. Rotary is not able to learn this at all.
- Small gains in sequence modeling of symbolic music, human genome and text.
- Competitive zero-shot performance on downstream tasks in text.
- Their method seems to extrapolate extremely well to extending the sequence length without re-training. This is probably the most interesting property in the paper.

**Comments.**
- I am somewhat suspicious about them doubling the dimension of the keys and query vectors. Although the phases are hardcoded, the increased dimension might be the cause of their small gains in their in-distribution evaluations.
- I wonder why the extrapolation to extended lengths is so good. When increasing the sequence length you naturally end up diluting attention between many tokens, even if the match is just residual noise. Does their method learn a very clean time decay? Could this harm long-range dependency problems in some way?
- There is no content-based angle filtering in the attention score. In other words, based solely on content all the scores would be positive. I am surprised that there is no loss in performance from this. I would have guessed that this makes it more difficult to separate important tokens from irrelevant ones.

