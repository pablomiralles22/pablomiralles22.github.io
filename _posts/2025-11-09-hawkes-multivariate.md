---
layout: post
title: Multivariate Hawkes process with Jax
description: Implementing the training of a multivariate Hawkes process efficiently with Jax
date: 2025-11-09
---

Recently, a colleague was giving a seminar on his ongoing work, and he was looking for an efficient implementation to train a multivariate Hawkes process. I though that it was a cool project to tackle with deep learning frameworks!
## Definition
A **multivariate Hawkes process** is a *self-exciting point process* that models how multiple types of events influence each other over time. Formally, we have $K$ event types, with baseline rates $\mu_i$, excitation matrix $\alpha_{ij}$, and exponential decay rate $\beta_{ij}$ (often shared as a single $\beta$). Each event type has a **conditional intensity** function:

$$
\lambda_i(t) = \mu_i + \sum_{j=1}^{K} \sum_{t_k^{(j)} < t} \alpha_{ij} e^{-\beta_{ij} (t - t_k^{(j)})}
$$

where $t_k^{(j)}$  are the times of past events of type j.
The intuition is that each past event of type $j$ adds an **exponential bump** to $\lambda_i(t)$ that:
* starts at height $\alpha_{ij}>0$ immediately after the event,
* decays continuously at rate $\beta_{ij}$,
* and is added on top of the baseline $\mu_i$.

The decay function can be substituted by any other function with similar properties, but in this implementation we assume the exponential function for convenience, as well as a single $\beta$ decay parameter. We also consider a low rank decomposition of the excitation matrix $\alpha_{ij} = u_i^T v_j$, or even $u=v$ if we assume symmetry.

## Training with maximum likelihood estimation
For observed events  ${(t_m, c_m)}_{m=1}^{N}$, where $t_m$ is the event time, $c_m \in {1,\dots,K}$ is the event type and $T$ is the observation horizon, the **log-likelihood** of a multivariate Hawkes process is:

$$
\log \mathcal{L} = \sum_{m=1}^{N} \log \lambda_{c_m}(t_m) - \sum_{i=1}^{K} \int_0^{T} \lambda_i(t) , dt
$$

With the exponential kernel the intensity functions is

$$
\lambda_i(t) = \mu_i + \sum_{j=1}^K \sum_{t_k^{(j)} < t} \alpha_{ij} e^{-\beta (t - t_k^{(j)})},
$$

and in this case the integral term has a **closed form**, avoiding numerical integration:

$$
\int_0^T \lambda_i(t) dt = \mu_i T + \sum_{j=1}^K \sum_{t_k^{(j)} < T} \frac{\alpha_{ij}}{\beta} \left(1 - e^{-\beta (T - t_k^{(j)})}\right).
$$

To train this model, you compute the log-likelihood above, and then maximize it (or equivalently, minimize the negative log-likelihood) with respect to the parameters:

$$
\Theta = { \mu_i, \alpha_{ij}, \beta }
$$

This can be done with any deep learning framework. We will be using Jax in this case instead of PyTorch, because I want to learn it and because it implements fast parallel scans out of the box.

## Derivations
We now consider the log-likelihood and try to make some derivations that make the vectorized implementation with Jax easy.

$$
\log \mathcal{L} = \underbrace{\sum_{m=1}^{N} \log \lambda_{c_n}(t_n)}_{\text{A: log-likelihood of observed events}} - \underbrace{\sum_{i=1}^{K} \int_0^{T} \lambda_i(t) \, dt}_{\text{B: integral over intensities}}
$$

**Derivation of the second term**

Let's consider the second term. First, since we have the exponential kernel:

$$
\int_0^T \lambda_i(t) dt = \mu_i T + \sum_{j=1}^K \sum_{t_k^{(j)} < T} \frac{\alpha_{ij}}{\beta} \left(1 - e^{-\beta (T - t_k^{(j)})}\right).
$$

Therefore, the second term becomes (considering $\alpha_{ij}=u_i^T v_j$):

$$
\begin{align*} B &= T \cdot \sum_{i=1}^K \mu_i + \sum_{i=1}^K \sum_{j=1}^K \sum_{t_k^{(j)}<T} \frac{\alpha_{ij}}{\beta}\!\left(1-e^{-\beta\,(T-t_k^{(j)})}\right) \\ &= T \cdot \sum_{i=1}^K \mu_i + \frac{1}{\beta} \sum_{i=1}^K u_i^T \left( \sum_{j=1}^K v_j \left[ \sum_{c_n=j} (1-e^{-\beta\,(T-t_n)}) \right] \right) \\ &= T \cdot \sum_{i=1}^K \mu_i + \frac{1}{\beta} \left(\sum_{i=1}^K u_i^T\right) \left( \sum_{n=1}^N v_{c_n} (1-e^{-\beta\,(T-t_n)})\right) \end{align*}
$$

This is simple to vectorize:

```python
v_mark = v[marks]  # (N, R)
summed_u = jnp.sum(u, axis=0)  # (R,)
exp_term = 1.0 - jnp.exp(-beta * jnp.clip(T - times, 0.0, None))  # (N,)
summed_v_mark = jnp.sum(v_mark * exp_term[:, None], axis=0)  # (R,)
B = jnp.sum(mu) * T + (1.0 / beta) * jnp.dot(summed_u, summed_v_mark)
```

**Derivation of the first term and naive scan**

The first term is a bit more difficult. Let's consider each $\lambda_{c_n} (t_n)$, and assume that the times $(t_n)_n$ are sorted:

$$
\begin{align*} \lambda_{c_n}(t_n) &= \mu_{c_n} + \sum_{j=1}^K \sum_{\substack{t_m<t_n \\ c_m=j}} u_{c_n}^T v_j \exp(-\beta(t_n-t_m)) \\ &= \mu_{c_n} + \sum_{m=1}^{n-1} u_{c_n}^T v_{c_m} \cdot \exp (-\beta(t_n-t_m)) \\ &= \mu_{c_n} + u_{c_n}^T \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta(t_n-t_m)) \end{align*}
$$

Now, can we parallelize this computation for all n? We can do the following next step:

$$
\begin{align*} \lambda_{c_n}(t_n) &= \mu_{c_n} + u_{c_n}^T \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta(t_n-t_m)) \\ &= \mu_{c_n} + \left( u_{c_n}^T \cdot \exp(-\beta t_n)\right) \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (\beta t_m). \end{align*}
$$

The sum term becomes a `cumsum` operation, which is a simple scan already implemented in many frameworks. However, the term $\exp(\beta t_m)$ can become very large, and this implementation could become numerically unstable.

**Non-naive scan**

Let's consider the quantity

$$
S_n = \sum_{m=1}^{n} v_{c_m} \cdot \exp (-\beta(t_n-t_m)),
$$

Then:

$$
\begin{align*} S_n &= v_{c_n} + \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta(t_n-t_m)) \\ &= v_{c_n} + \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta((t_n-t_{n-1}) + (t_{n-1}-t_m))) \\ &= v_{c_n} + \exp(-\beta(t_n-t_{n-1}))\sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta(t_{n-1}-t_m)) \\ &= v_{c_n} + \exp(-\beta(t_n-t_{n-1})) \cdot S_{n-1} .\end{align*}
$$

This can be seen as a reduction for the operator

$$
\begin{array}{rcl} *: & (\mathbb{R}^d \times \mathbb R) \times (\mathbb{R}^d \times \mathbb R) & \longrightarrow \mathbb{R}^d \times \mathbb R \\[3pt] & (v_1, t_1),(v_2,t_2) & \longmapsto \left( v_2 + \exp(-\beta(t_2-t_1)) \cdot v_1 , t_2\right), \end{array}
$$

which is associative, and this can be implemented as a parallel scan, like the cumulative sum. The difference now is that since the timestamps are ordered, the quantities $\exp(-\beta (t_2-t_1))$ with $t_2>t_1$ are always less than $1$, and they are bounded and controlled. We can now derive the final implementation:

$$
\begin{align*} \lambda_{c_n}(t_n) &= \mu_{c_n} + u_{c_n}^T \sum_{m=1}^{n-1} v_{c_m} \cdot \exp (-\beta(t_n-t_m)) \\ &= \mu_{c_n} + u_{c_n}^T (S_n - v_{c_n}) \end{align*}
$$

The problem now is that this scan is not implemented natively in PyTorch, and that is why we are using Jax:
```python
u_mark = u[marks]  # (N, R)
v_mark = v[marks]  # (N, R)
mu_mark = mu[marks]  # (N,)

def _scan_op(e1, e2):
    (u1, t1), (u2, t2) = e1, e2 # each is (R,), scalar
    decay = jnp.exp(-beta *(t2 - t1))
    return (u2 + decay[:, None] * u1, t2)
scan_vals, _ = lax.associative_scan(_scan_op, (u_mark, times), axis=0)

lambdas = mu_mark + jnp.sum(u_mark * (scan_vals - v_mark), axis=1)
A = jnp.sum(jnp.log(lambdas))
```

## Complete code
The full code with a training example is provided at [pablomiralles22/multiprocess-hawkes-jax](https://github.com/pablomiralles22/multiprocess-hawkes-jax).
