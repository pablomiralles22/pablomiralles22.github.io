---
layout: post
title: Running LLMs in low-resource settings
description: A small index of all ideas I have come to learn to run LLM efficiently, and with low resources.
date: 2025-11-19
---

In this post I try to create an index of the main ideas I have found to be able to run Large Language Models in my low-resource servers, and to run them as efficiently as possible. The point is not to explain every technique in-depth, but to provide an index with high-level descriptions. I have decided to divide them in three categories: **general**, **training-specific** and **inference-specific** optimizations.

## General
#### FlashAttention
FlashAttention is an algorithm to compute

$$
\mathrm{Attention}(K, Q, V) = \mathrm{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

efficiently in GPUs. As you probably know, this formula is quadratic in the input sequence length, making it problematic with very long sequences. FlashAttention remains quadratic in execution time, but avoids materializing the full $N \times N$ attention matrix in memory by computing the result in small tiles of $Q$, $K$ and $V$, and recomputing the result on the backward pass for training instead of storing the attention matrix. New versions with additional optimizations are available, but I think these are the most important one. We do not need to worry about setting this algorithm because all modern models use FlashAttention by default.

#### Mixture-of-experts (MoE)
A mixture-of-experts (MoE) model splits a neural network into many specialized _experts_, and uses a **gating network** to decide which experts handle each input. In the case of LLMs, this is typically applied in the feed-forward layers. Instead of scaling the MLP layers by widening them, the network uses several MLP experts and routes each token to a few of them (usually 1-2). This scales the number of effective parameters without increasing inference costs too much.

As a side note, during training we must make sure that the tokens are spread across experts to avoid collapsing to a subset of experts. This can be done via an auxiliary load-balancing loss term, adding noise to the gating network that routes token, or other ideas.

#### Data/model parallelism
The concept of parallelism is key both to speed up execution and to be able to run large models that do not fit in a single GPU.

The simplest type is **data parallelism**, where we store a full copy of the model in each GPU, and the data is sharded across GPUs and processed independently. For training it is then necessary to sync the gradients in all devices before performing the backward pass. This option is great to speed up execution, but it requires being able to fit the model in each device.

In **model parallelism**, each device holds only _part_ of the model, and the execution of the model is orchestrated across them. We can distinguish two types:
- **Tensor parallelism**. Each layer of the model is split across the devices. For example, the columns or rows in the weight matrix of a linear layer could be divided by GPU, and the result synced and concatenated between devices.
- **Pipeline parallelism**. The layers are grouped in stages and divided into the devices. For example, the first device could execute layers 1 to 4, and the second device layers 5 to 8. Of course, there is a dependency between the devices, both in the forward and in the backward pass. To avoid idle time, the batch is divided in micro-batches and the execution segmented across them. For example, the first device could execute the first micro-batch, and pass the result to the second device. While the second device processes the first micro-batch, the first device would start processing the second micro-batch.

There are several advantages and disadvantages for both options. Tensor parallelism enables very big layers to be run and doesn't need micro-batching, but it needs a lot of communication between devices, requiring high-bandwidth/low-latency interconnects (e.g. NVLink). On the other hand, pipeline parallelism needs minimal communication between devices, but requires micro-batching and balanced stages to avoid idleness.

All types of parallelism discussed thus far are depicted in the following figure:
{% include figure.liquid path="assets/img/posts/2025-11-19-llm-efficiency/parallelism.drawio.svg" class="img-fluid rounded z-depth-1" %}

One final note is that all of these types of parallelism can be combined. For example, we could have two copies of the model, each one using three devices via pipeline parallelism, requiring six GPUs in total.

Regarding the implementation, for inference I recommend the `vLLM` library, which allows the use of all types of parallelism via simple command line arguments. Training is a bit more complex. The `transformers` library offers data parallelism, as well as the `device_map="auto"` option when loading the model, which implements some rudimentary pipeline parallelism without segmentation (thus underutilizing GPUs greatly). For a more optimized or complete optimization, we would have to use something like the `deepspeed` library.

#### Lower precision, mixed precision and quantization
Most operations can be run in 16-bit precision (FP16 or BF16) without problems, allowing us to reduce memory usage by a factor of 2, and speeding up computations a lot with modern hardware that is adapted to use this precision.

Some operations like reductions/accumulations or normalizations are better performed in FP32 to avoid overflow/underflow or to reduce rounding errors. Thus, it is common to keep these operations in 32-bit precision (**mixed precision**). During training, it is also common to keep master weights of the model or optimizer states in FP32 as well.

A step further is **quantization**, representing model parameters and sometimes activations with integers (e.g., int8, int4) instead of floating point (FP32/BF16/FP16). The goal is to reduce memory footprint and improve inference speed, especially on hardware that has fast integer arithmetic. The trick here is performing this compression of the model without losing too much accuracy. Very often, we find versions of the top open-source models already quantized with modern techniques like AWQ, so we do not need to worry about it.

#### CPU offloading
If you are very short on memory but absolutely need to run a large model, you can offload some of the weights to CPU, and swap them around with the GPU to complete all computations. However, I highly recommend against it because it is extremely slow.

## Training-specific
#### Parameter efficient fine-tuning: LoRA and soft prompting
In language modeling, we typically start from a pre-trained model that is then fine-tuned in our downstream task of interest. In most cases, it is sufficient to perform a "reduced" update to the network instead of updating all parameters, greatly reducing memory requirements for the gradients and optimizer states. What is more, given the small amount of labeled data we often have and the high dimensionality of textual data, these reduced updates are more likely to generalize better.

The most common example is **LoRA (Low-Rank Adaptation)**, where a matrix $W \in \mathbb R^{n_{in} \times n_{out}}$ is updated with a low rank matrix $W_{new} = W + AB$ with trainable matrices $A \in \mathbb R^{n_{in} \times r}$ and  $B \in \mathbb R ^{r \times n_{out}}$ and a small rank $r$. Another example is the concept of **soft prompting**, where a few trainable embeddings are prepended to all input sequences.

#### QLoRA (Quantized Low-Rank Adaptation)
The next step in LoRA is to quantize the base model while training non-quantized low rank adapters, achieving very good compression without sacrificing a lot of performance.

#### Optimizers with smaller states
The Adam optimizer uses $2N$ values for the momentum and second moment running estimates, where $N$ is the number of trainable parameters in the model. We can reduce it to $N$ by switching to Nesterov or RMSProp, or reduce it event further to $\approx\sqrt{N}$ with Adafactor. The tradeoff in all cases is reducing training stability and convergence speed.

#### Accumulate gradient batches
Instead of using a large batch size, we can instead use a small batch size and accumulate the gradient for multiple batches before performing an optimization step. This is equivalent thanks to the linearity of the gradient operator, except in models that include batch level operations (e.g. batch normalization), which is rare in language models. With this, we trade-off training speed to reduce memory usage, which is often the stricter limiting factor.

#### Gradient checkpointing
During the forward pass, it is common to store values that will be used in the backward to avoid wasteful recomputation. However, this is undesirable when the bottleneck is the GPU memory available. Disabling this is called **gradient checkpointing**.

#### Gradient/optimizer state sharding
While training in a data parallel setting, each GPU contains a copy of the gradients and optimizer states for all parameters. However, this is a bit wasteful, and we can shard (divide) these vectors across GPUs at the expense of some communication overhead.

## Inference-specific
#### KV Cache
Autoregressive models use a causal attention mask that blocks each token from attending to future tokens. As a result, when generating text one token at a time, newly produced tokens don’t influence the earlier ones, only the other way around. This means we don’t need to redo computations for previous tokens. However, the new token still depends on earlier tokens through their attention keys and values. To avoid recomputing these for every step, we store the keys and values of all previously generated tokens in a **KV cache**. During each new forward pass, the model retrieves these cached representations and only performs computations for the new token, greatly speeding up generation. Regarding the implementation, this is already done in the typical generation libraries (e.g. `HuggingFace` or `vLLM`).

#### Paged attention
**Paged attention** implements a type of paged virtual memory for the KV cache in the GPU RAM, removing the need for large contiguous blocks for long sequences and improving memory utilization. This is implemented in `vLLM`.

#### Speculative decoding and draft models
Speculative decoding pairs a large language model with a smaller, faster "draft model". The draft model predicts several upcoming tokens, and the large model then checks those predictions in parallel, accepting the ones that match what it would have produced itself and discarding the rest. This allows the system to generate multiple tokens per verification step instead of one at a time, significantly reducing latency while preserving the large model’s output quality.

#### Efficient batching and scheduling
`vLLM` implements and efficient scheduling algorithm where new requests made to the model are dynamically added to the current batch, and completed generations are dynamically removed from the batch as well, improving resource utilization.

#### LoRA for multi-model serving
When serving the same base model with several LoRA adapters on top, you can share the common base in GPU memory and enable or disable the pertinent adapters at runtime, making it very efficient to deploy. This is also implemented in `vLLM`.
