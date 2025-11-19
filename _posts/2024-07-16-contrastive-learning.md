---
layout: post
title: An introduction to contrastive techniques for representation learning
description: A look at some of the basic research on contrastive learning
date: 2024-07-16

tags:
- deep-learning
- contrastive-learning
---

Earlier this year I had to write a technical report on the topic of contrastive
learning
([link](/assets/pdf/projects/contrastive-learning-report/report.pdf)).
This post is an effort to create something more digestible and adequate for
reading. The goal is to give an introduction to representation learning. This
is done in the first section, also discussing its advantages and other related
topics such as self-supervised or transfer learning. The second section is
dedicated to the techniques for representations learning labelled as
"contrastive". We will go over some of the most important topics: loss
functions, data generation techniques, negative examples and applications.

## Representations, transfer learning and self-supervised learning

Consider some neural network such as the following:

{% include figure.liquid path="assets/img/posts/2024-07-16-contrastive-learning/nn_repr.pdf" class="img-fluid rounded z-depth-1" %}

Neural networks are the composition of transformations, where every
intermediate layer applies a parameterized function to the input coming from
the previous one. The output of each intermediate layer is a new view or
**representation** of the input data. These representations are nothing but
vectors, where the values are features or characteristics of the input. Of
course, neural networks are not interpretable, and these vectors do not
necessarily hold any meaning for us. The network learns to progressively
transform the data into a set of features that is useful to perform the task we
are training it for. For example, if we are training for image classification
of animals, the network might learn a representation with the number of legs,
the shape of the eyes, the color...

At the end of the network, we have a final layer that transforms the last
representation into the prediction, such as a regression value or a probability
distribution over the different classes in our classification problem.

When we train a network on one dataset, we are not only teaching it how to make
predictions, we are teaching it to extract intermediate representations with
features that are useful for the task. This is interesting because these
representations might be useful for other tasks as well. For example, imagine
we want to add a new class of animals in our classification example. Perhaps
the network is already capable of getting features to predict that class. We
can then modify the final head and freeze the rest of the weights of the
network. With fewer parameters to learn, we can train with less data, much
faster and with a smaller energy consumption. In fact, if the learned
representations are rich enough, we might be able to use the same network for a
myriad of different tasks, training only a small head at the end of the
network. This is called **transfer learning**: first **pretraining** for a
**pretext task**, and then **fine-tuning** the model (perhaps modifying the
head) on the **downstream task** we are actually interested in solving. The
following diagram shows this process.

{% include figure.liquid path="assets/img/posts/2024-07-16-contrastive-learning/transfer-learning.drawio.pdf" class="img-fluid rounded z-depth-1" %}

### Learning very general representations: self-supervised learning

Another important and related concept is that of **self-supervised learning**.
This term, although poorly coined in my opinion, can be useful to refer to a
particular set of tasks. It refers to supervised tasks where we can generate a
training signal automatically, without the need for human labeling. One example
is the training procedure of the language model BERT<d-cite
key="devlin2019BERTPretraining"/>, called Masked Language Modeling (MLM). In
this task, some parts of the input text are masked, and the model has to
recover them based on the unmasked context. We can then use any available text
on the internet and generate training examples by masking parts of it. This
allows us to generate massive datasets, and thus train very large neural
networks with a nuanced understanding of the distribution of human text. These
large networks learn very general representations, that we can then transfer to
a great number of tasks. In fact, BERT achieved state-of-the-art performance in
several downstream tasks.

Let's consider another example, this time with images. Images allow us very
simple forms of data augmentation: we can crop, rotate, translate, convert to
greyscale, modify the brightness... We can train a neural network to produce
similar representations for augmented views of the same image, and different
representations for views of different images. Again, we can generate examples
with any image we find, allowing us to create very large datasets and train
massive networks that obtain general representations. As we will see, this is
our first example of contrastive learning.

### Advantages and disadvantages

Representation learning, together with transfer learning and self-supervised
learning, has produced a revolution in the field. Some of the benefits we get
from these techniques are:

1. **It can greatly improve performance for some tasks**. Let's consider the
   example of sentiment analysis in text. Textual data is very diverse and
   nuanced. Even the same word can represent different sentiments depending on
   the context. A small dataset of examples labeled by humans is unlikely to
   produce a model with detailed understanding of text. By pretraining a large
   model on a difficult task with a gigantic set of examples, we get a model
   with nuanced understanding of text, and this is likely to produce better
   results on our downstream task as well.

2. Another way to put the previous point is that **it alleviates data
   bottlenecks**. If we don't have enough data to train a decent model for a
   task, we can try to pretrain on some pretext task and then fine-tune small
   parts of the model with the supervised data we have.

3. **It improves generalization**. Let's consider the sentiment analysis task
   again. A pretrained model learns representations for many different tokens
   in a lot of different contexts, since the dataset is very large. In this
   common feature space, we may find specific features about sentiment, which
   may be used by the final head we train on supervised data.

   Since the feature space is common to a large number of tokens and contexts,
   even if they do not appear on the supervised data, the model will be able
   to extrapolate what they learn.

4. **It allows us to use large and powerful models at a fraction of the cost**.
   We find many open source models nowadays, and using them for inference
   requires much fewer resources than for training.

5. **Some model have zero-shot capabilities**. This means that we can perform
   inference on tasks it wasn't trained for without any need for fine-tuning.
   We have all seen ChatGPT performing a myriad of different tasks, even though
   it was only trained to do next-token prediction. Another example is
   CLIP {% cite radford2021LearningTransferable --file 2024-07-16-contrastive-learning %}. This model was trained
   using contrastive techniques. It encodes text and images into a common
   feature vector space. If a text describes an image properly, then their
   representation should be close together in the space. This allows us to
   perform arbitrary image classification as long as we have textual labels for
   the classes. We simply pick the class with the textual label that is closest
   to the image embedding.

Of course, not everything is bright. Training big and general models is very
expensive, and only big companies with large resources can afford to do so. We
may also find that these models are overkill for some tasks. A smaller and more
specialized model can sometimes perform better, specially if the learned
representations aren't very good for the particular task. Finally, although
zero-shot capabilities are very interesting, performance tends to be subpar.


## Contrastive learning

Roughly speaking, in contrastive learning we try to learn vector representation
of the input data by comparing and contrasting examples with each other. For
instances that we deem similar in some sense, we want their vector
representation to be close according to some distance or similarity function.
On the other hand, the vector representation of dissimilar examples should be
further apart. Images provide an easy example. We would expect pictures of
animals to be close together, and far apart from pictures of buildings. Inside
the cluster of pictures of animals, we would also like for pictures of the same
animal to be closer than pictures of different animals, and so on.

### Basic concepts and notation

Let’s establish a framework and notation for the rest of the text.

- We first consider a set of networks, called *encoders*, that map the input
  data to a fixed sized vector representation of dimension $d$. We may have
  one or multiple encoders, depending on the problem. For example, if we are
  trying to map both text and images to a common representation space, we
  need a different encoder for each domain.

  Mathematically, the $k$-th encoder $e_k$ maps the examples in some input
  space $\mathcal{X}_k$ onto a $d$-dimensional vector space:

  $$
    \begin{align*}
	e_k(\cdot;\theta_k): \mathcal{X}_k &\longrightarrow \mathbb{R}^d \\\\
	x &\longmapsto e_k(x; \theta_k )
    .\end{align*}
  $$

  Here, $\theta_k$ are the parameters of the neural network that we would like
  to optimize. In cases where a single encoder is used, we will omit the $k$
  subscript.

- Additionally, we can assume without loss of generality that a final
    head $h_k(\cdot, \eta_k)$ maps these representations to an $m$-dimensional
    metric space, where the notion of distance or similarity is actually
    applied between the vectors. The point of adding this final transformation
    is to separate two tasks: finding a general representation that can be applied
    to other problems and finding a representation in a metric space where the
    notion of distance represents some real-world semantic similarity. The former
    is typically better for transfer learning. If we didn't need this separation,
    we may assume that this is the identity function.

- The metric is given by a distance or similarity function $\mathbb{R}^m
    \times \mathbb{R}^m \to \mathbb{R}$, such as the Euclidean distance
    $\|\|z_1 - z_2\|\|$ or the cosine similarity function $\frac{\langle z_1,
    z_2 \rangle}{\|\|z_1\|\| \|\|z_2\|\|}$. Similarity functions take larger
    values for similar examples, and will be denoted by $S$, whereas distance
    functions, denoted by $S$, are lower bounded by zero and take smaller
    values for similar instances.

- A loss function $\mathcal{L}$ is built using this similarity notion
    for a set of encoded examples, penalizing large distances for similar
    instances, and small distances for dissimilar examples. We will take
    a closer look into loss functions in the next section.

- Finally, given an example $x \in \mathcal{X}$, we will denote
    its vector representation in the general feature space by $v=e(x)$, and its
    embedding in the metric space by $z=h(v)$. Instances that we consider
    similar to an anchor example $x$ will be called positive examples and be
    denoted by a plus superscript $x^+$. We define negative examples
    analogously and denote them by a minus superscript $x^-$. The same
    superscript notation will also be applied to vectors $v$ and $z$.

This structure is shown graphically for two examples in the following diagram:

{% include figure.liquid path="assets/img/posts/2024-07-16-contrastive-learning/contrastive-learning.drawio.pdf" class="img-fluid rounded z-depth-1" %}

### Loss functions in contrastive learning

Loss functions are generated from the distance/similarity metric applied to
different pairs of examples. It should penalize similar examples being far away
and dissimilar examples being close. The latter is as important as the former,
otherwise the network could learn the trivial representation of mapping any
example to the same vector. Let's take a look at some common functions.

#### Pair loss

Let $x \in \mathcal{X}$ be an example and $z$ its embedding in the metric
space. The *pair loss* {% cite chopra2005LearningSimilarity --file 2024-07-16-contrastive-learning %} is defined
for pairs of examples, differing for positive and negative ones: 

$$
  \begin{cases}
    \mathcal{L}(x, x^+) &= D(z, z^+)^2 \\
    \mathcal{L}(x, x^-) &= \max (0, \varepsilon - D(z, z^-)^2),
  \end{cases}
$$

where $D$ is some distance function such as the Euclidean distance. By
minimizing this expression, the distance to positive samples should be close to
zero, and dissimilar instances should be separated by a margin of at least
$\varepsilon$. The function is plotted in the following figure.

<!-- <figcaption>
    Plot of the pair loss function for positive (left) and negative (right)
    examples. Arrows represent gradient direction and norm.
</figcaption> -->

{% figure [caption:"Plot of the pair loss function for positive (left) and negative (right) examples. Arrows represent gradient direction and norm."] [class:"class1 class2"] %}
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; margin-top: 20px;">
    <div style="flex: 1; text-align: center;">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-07-16-contrastive-learning/pair_loss_pos.pdf" zoomable=true %}
    </div>
    <div style="flex: 1; text-align: center;">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-07-16-contrastive-learning/pair_loss_neg.pdf" zoomable=true %}
    </div>
</div>
{% endfigure %}


#### Triplet loss

The *triplet loss* {% cite schroff2015FaceNetunified --file 2024-07-16-contrastive-learning %} tries instead to
enforce that the distance between the anchor and positive examples is smaller
than the distance between the anchor and negative examples, with a difference
margin of at least $\varepsilon$:

$$
\label{eq:triplet-loss}
\mathcal{L}(x, x^+, x^-) = \max(0, D(z,z^+)^2 - D(z,z^-)^2 + \varepsilon)
.$$

In this last loss function, a full training example requires now both a similar
and a dissimilar instance. However, the number of interactions between
different examples is still very limited. The use of many negative examples,
specially those that are hard for the model, are crucial for a good and
efficient learning process. This makes sense intuitively: there are many more
ways in which images can be different than similar.

#### Lifted Embedding Loss

Increasing the number of interactions between instances at once, Song et al.
proposed the *Lifted Embedding loss* {% cite song2016DeepMetric --file 2024-07-16-contrastive-learning %}. If
$x_1, \cdots, x_n$ is our set of examples, and $P$ and $N$ are the sets of pairs
of examples that are considered similar and dissimilar, respectively, then the
loss function is given as

$$\label{eq:lifted-structure-loss}
  \mathcal{L} (N, P) = \frac{1}{2 |P|} \sum_{(i,j) \in P} \max (0, L_{i,j})^2
,$$

where

$$L_{i,j} = D_{i,j} + \log \left( 
    \sum_{(i,k) \in N} e^{\varepsilon - D_{i,k}} +
    \sum_{(j,l) \in N} e^{\varepsilon - D_{j,l}}
   \right)
,$$

and $D_{i,j} = D(z_i, z_j)$. The expression for $L_{i,j}$ is
actually used because it is a smooth upper bound for

$$\hat{L}_{i,j} = D_{i,j} + \max \left( 
    \max_{(i,k) \in N} \varepsilon - D_{i,k},
    \max_{(j,l) \in N} \varepsilon - D_{j,l}
   \right)
,$$

so this loss can be interpreted as an adaptation to the triplet loss, trying to
mine the hardest negative example of the set for each positive pair, and also
squaring the result. As usual in deep learning, instead of using the full set
of examples, the loss is approximated with a batch of smaller size.

#### Binary Noise-Contrastive Estimation (NCE) loss

Let’s now take a probabilistic perspective. Let $X_1$ and $X_2$ be two
random variables that take values in our domain of examples $\mathcal{X}$
(e.g. images). Let $Y$ be another random variable that takes the value of
$1$ if $X_1$ and $X_2$ are similar according to our concept of similarity,
and $0$ otherwise. It follows a Bernoulli distribution, and we can try to
approximate its conditional probability mass function with our network as

$$
\hat{p} (1 | x_1, x_2) = \sigma ( S(z_1, z_2) )
,$$

where $\sigma$ is the sigmoid function. If $q^+(\cdot, \cdot)$ and $q^-(\cdot,
\cdot)$ are the joint probability density functions of similar and dissimilar
instances, respectively, then the Binary Noise-Contrastive Estimation
(NCE) {% cite gutmann2010Noisecontrastiveestimation --file 2024-07-16-contrastive-learning %} loss is given by
minimizing the expected negative log-likelihood:

$$\begin{gathered}
  \label{eq:bin-nce}
  \mathcal{L}_{Bin-NCE} =
  -\mathbb{E}_{q^+} \log \hat{p} (1| x_1, x_2) \\
  -\mathbb{E}_{q^-} \log (1-\hat{p} (1| x_1, x_2))
,
\end{gathered}$$

where the expected value would be approximated by its population mean (e.g.
with a single batch). Keeping the previous notation for the sets of positive
pairs $P$ and negative pairs $N$, this yields:

$$\begin{gathered}
  \label{eq:bin-nce-2}
  \mathcal{L}_{Bin-NCE} =
  -\frac{1}{|P|} \sum_{(i,j) \in P} \log \sigma(S(z_i, z_j))  \\
  -\frac{1}{|N|} \sum_{(i,j) \in N} \log (1-\sigma(S(z_i, z_j)))
.
\end{gathered}$$

#### InfoNCE

A more recent loss function is InfoNCE <d-cite
key="oord2019RepresentationLearning"/>. In their setting, instead of
considering a binary classification problem they assume a ranking one. Having
fixed an anchor instance $x$, let $x_0^+,x_1^-, \cdots,x_n^-$ be a set of
possible similar instances, where only $x_0^+$ is a positive example. The
problem becomes one of ranking which example is more likely to be positive. The
probability of each of the samples can be approximated by a softmax operation
on a similarity score with respect to $x$:

$$
\label{eq:info-nce-p}
\hat{p} (i|x) = \frac{\exp(S(x, x_i))}{\sum_{j = 0}^{n} \exp(S(x, x_j))}
.$$

Minimizing the negative log-likelihood of the true positive yields:

$$
\label{eq:info-nce}
\mathcal{L}_{InfoNCE} = 
  - \mathbb{E} \log \frac{\exp(S(x, x_0^+))}{\sum_{j = 0}^{n} \exp(S(x, x_j))}
.$$

As we will see, it is a common setting to have batches of pairs of similar
instances $(x_1,x_1^\prime),\cdots,(x_n,x_n^\prime)$, where instances across
pairs are considered to be dissimilar. We can compute a similarity matrix
$\mathcal{S}  = (S(z_i, z_j^\prime))_{i,j}$, where the main diagonal values
should be high and the rest should be low. In this case, one can calculate the
InfoNCE across rows or columns. It is also possible to average both options,
yielding a *symmetric InfoNCE* loss.

#### NT-Xent

A temperature parameter $\tau$ can be included in the sotfmax operation,
transforming the softmax probability distribution into

$$
P(i|x) = \frac{\exp(S(x, x_i) / \tau)}{\sum_{j = 0}^{n} \exp(S(x, x_j) / \tau)}
.$$

A small value of $\tau$ makes the softmax sharper, and small differences
between the similarity of positive and negative examples already produce a
high likelihood. A large value of $\tau$ forces the difference in similarity to
be large. This parameter can be viewed as the margin parameter in previous
functions. This modified InfoNCE loss is called *NT-Xent* (normalized
temperature-scaled cross entropy loss) {% cite chen2020SimpleFramework --file 2024-07-16-contrastive-learning %}.

### Generating data for contrastive learning

We have already seen how to create a loss function to train our model, but how
do we get appropriate data? In most cases, the question is how to generate
pairs of *similar* examples, as you can often get away with the assumption that
examples in different pairs are dissimilar. This is the case when you have a
huge domain, such as a large amount of internet images. Getting two similar
examples in the same batch by sampling randomly is unlikely. Still, with this
assumption we could incur in false negatives, something we will discuss in the
next section. Let's see now some of the ways in which datasets are generated
for contrastive learning.
<!-- This setting is called *instance-level discrimination*<d-cite -->
<!-- key="wu2018UnsupervisedFeature"/>, as we assume that each instance forms its -->
<!-- own class. We will now list some of the ways in which we can generate datasets -->
<!-- for contrastive learning. -->

#### Human supervision

Of course, human annotation is always an option, albeit a very expensive one.
Unfortunately, this is sometimes necessary. We previously talked about CLIP, a
model that learned representations for text and image, mapping a text that
correctly describe an image to an embedding that is close to the embedding of
the image. In this case, there is no way around getting someone to create
captions for different images (although in the future synthetic data generation
with multimodal models might be an option).

One way to reduce the amount of data necessary when human labeling is required
is to first pretrain the encoders with some other self-supervised technique.
For example, in CLIP, you could train the text and image encoders separately
with a denoising task, and then use the supervised data to align the
representations of both encoders to a common metric space.

#### Self-supervision

What if we want to generate large amounts of data from existing sources
automatically? As explained earlier, we will generate pairs of similar
instances, and assume that images in different pairs are dissimilar. The
following are just some examples:

* **Data augmentation**. In the particular context of contrastive learning, a
    transformation that does not change the instance semantically can be applied to
    generate positive examples. To illustrate this, assume that we are
    contrasting images, and we want to map images based on the concepts inside
    them. Then if we crop, blur, or saturate the color of an image of a cat, it
    is still an image of a cat. This way, augmentations of similar or equal
    instances are also similar, whereas augmentations of dissimilar examples
    are also dissimilar.

    In the case of images, we find transformations such as rotations,
    translations, cutouts, cropping and resizing, blurring, applying noise...
    These were used, for example, in {% cite chen2020SimpleFramework --file 2024-07-16-contrastive-learning %}.

    Textual data is a bit more complex. Fang et al.<d-cite
    key="fang2020CERTContrastive"/> use automatic back-translation, that is,
    translation to a different language and then back to the source language,
    to generate pairs of different but semantically equivalent segments of
    text. Many other techniques can be used: changing characters, masking
    words, adding noise, changing words for synonyms...

    Careful exploration of data augmentation techniques can be very important.
    For example, Chen et al. {% cite chen2020SimpleFramework --file 2024-07-16-contrastive-learning %} found that
    the training of SimCLR was very sensitive to the choice of data
    augmentation techniques.

* **Multi-sensor**. When multiple inputs are captured simultaneously, we can
    use all inputs from corresponding times. For example, capturing the same
    image from different angles with multiple cameras, or the audio and images
    in videos.

* **Continuity**. This can be applied to domains where there
    are sequences with some continuity. For example, videos are sequences of
    images where most contiguous frames are almost equal. We can take images
    very close in time and assume they are very likely to be similar.


### Discussion on negative examples

The loss functions we have seen thus far use both positive and negatives
examples. We already gave a theoretical reason for this: if negative examples
were not used, a trivial representation mapping everything to one vector would
obtain perfect performance. There is empirical evidence showing that
performance is increased from contrasting with many negative examples. For
example, Chen et al. {% cite chen2020SimpleFramework --file 2024-07-16-contrastive-learning %} found that training
with very large batch sizes greatly improved performance. In their setting, all
other examples in the batch were considered as negative, and thus the number of
negatives per example scaled with the batch size.

The concept of negative examples or, more generally, preventing
representational collapse, is quite important in contrastive learning, and
there are some topics around it that are worth discussing.

#### False negatives in self-supervised contrastive learning

In the data generation section we already talked at the possibility of false
negatives when there is no supervision. This is because we are drawing from the
full distribution of examples instead of the distribution of negative examples,
creating a bias. Some work has been done on alleviating this while not requiring
manual labels for negatives. While we won't get into detail, the reader may
find an example in the *Debiased Contrastive loss* proposed in <d-cite
key="chuang2020DebiasedContrastive"/>.

#### Alleviating hardware bottlenecks for large amounts of negative samples

The use of examples in the same batch as negatives presents
a major drawback in terms of computing resources. If we want to use many
negative examples, we are forced to select a very large batch size, requiring a
lot of GPU memory. Some work has been done on decoupling batch size and the
number of negatives by sampling negatives from an *offline memory bank*. In
this setting, an encoded representation is kept on-disk for some or all
examples, and the loss function is not back-propagated through them. As the
encoder is updated in the training process, the representations on-disk get
outdated. The main difference between approaches lies in the method to keep
these offline representations updated as the encoder is optimized.

Wu et al. {% cite wu2018UnsupervisedFeature --file 2024-07-16-contrastive-learning %} sampled negative
representations randomly from a memory bank with the full dataset. At the end
of each epoch, all the representations in the memory bank are updated with the
new checkpoint of the model.

He et al. {% cite he2020MomentumContrast --file 2024-07-16-contrastive-learning %} proposed keeping a queue with
a fixed number of mini-batches. After processing a mini-batch, the new examples
are added to the queue, and the oldest mini-batch is removed. The queue is used
to sample negative examples for the current mini-batch. Since this alone
resulted in poor empirical performance, they separated the online encoder that
is being trained from an offline encoder that produces the representations for
the queue. The parameters of the offline encoder are updated through a momentum
update rule with the parameters of the online one:

$$\label{eq:moco} 
  \theta_{off} \leftarrow 
    \alpha \theta_{off} + (1-\alpha) \theta_{on},
  \qquad \alpha \in [0,1)
.$$

This smoother update yielded better empirical performance.

#### Hard negative mining

While increasing the number of negative examples has been observed to
improve performance, this might be due to the increased probability of
finding meaningful negatives to learn from.

Conceptually, we might intuit that it is more difficult for a model to
distinguish between a dog and a cat than between a dog and a building. In
practice, we could select hard negative examples based on their representation.
Examples with close representations are difficult to distinguish for the model.
Kalantidis et al. {% cite kalantidis2020HardNegative --file 2024-07-16-contrastive-learning %} make use of this,
together with some data mixing techniques.

However, hard negative mining has some disadvantages, such as the
increased time complexity from sampling close neighbours and the
increased probability of drawing false negatives in the self-supervised
setting as the encoder gets better.

#### Are negative examples really necessary?

If the only reason to use negative examples is to prevent the
representations from collapsing onto one single vector, then other
techniques to prevent it could allow us to remove negative examples.

The work by Grill et al. {% cite grill2020BootstrapYour --file 2024-07-16-contrastive-learning %} goes in this
line. They used two neural networks, an online (predictive) network and a
target network, similar to Deep Q Learning. They use only positive examples,
and for each pair the online network tries to predict the metric representation
from the target network. The parameters of the target network are updated after
every iteration with an exponential moving average of the online parameters:

$$
\theta_{target}  \gets \alpha \theta_{target}  + (1 - \alpha) \theta_{online} 
.$$

It is not obvious to me that this approach would not converge to a collapsed
representation. The authors argue that the update to the target parameters is
not exactly according to the gradient of the loss with respect to
$\theta_{target}$. They did get good performance in downstream image
classification tasks. There is some discussion on informal mediums (see 
[this blog post](https://imbue.com/research/2020-08-24-understanding-self-supervised-contrastive-learning/)
and preprint {% cite richemond2020BYOLworks --file 2024-07-16-contrastive-learning %}) on whether batch
normalization is the cause of preventing representational collapse, but I
haven’t found peer-reviewed work on the topic.

SimSiam {% cite chen2021ExploringSimple --file 2024-07-16-contrastive-learning %} is an even simpler technique
that successfully avoided representational collapse. I deviate from their
notation in the explanation to keep the one in this post. If $\hat{e}=e \circ
h$ is the complete encoder onto the metric space, and $\hat{h}$ is an additional
head they call the predictor, then their loss functions for similar instances
$x_1$ and $x_2$ is given by the formula:

$$
z_1 = \hat{e} (x_1),\quad z_2 = \hat{e} (x_2)
$$

$$
p_1 = \hat{h} (z_1),\quad p_2 = \hat{h} (z_2)
$$

$$
\mathcal{L} = \frac{D(p_1, SG(z_2))}{2} + \frac{D(p_2, SG(z_1))}{2}
,$$

where $SG$ is the stop-gradient function, preventing backpropagation through
that branch of computation, and $D$ is the negative cosine similarity. Again,
how this asymmetry through the additional head and stopping gradient
propagation achieves its objective of avoiding representation collapse is not
fully understood (to the best of my knowledge).

Bardes et al. {% cite bardes2022vicreg --file 2024-07-16-contrastive-learning %} proposed a much more intuitive
approach, using VICReg (Variance-Invariance-Covariance regularization). Let
$(z_1,z_1^\prime),\cdots,(z_n,z_n^\prime)$ be $d$-dimensional metric
representations of positive pairs in a single batch. Let $C(Z)$ be the
covariance matrix **of the features** for the matrix $Z=(z_1,\cdots,z_n) \in
\mathbb{R}^{n \times d}$, that is,

$$
C(Z) = \frac{1}{n-1} \sum_{i=1}^{n} (z_i - \overline{z})(z_i - \overline{z})^T
,$$

where $\overline{z} = \frac{1}{n} \sum_{i=1}^{n} z_i$ is the empirical mean. Then the
loss function has the following three terms:

* **Invariance**. Forces the positive examples to be close together:

    $$
    \mathcal{I} = \frac{1}{n}\sum_{i=1}^{n} \| z_i - z_i^\prime \|^2
    .$$
* **Variance**. Forces the variance of each feature to be positive, above some
    threshold value:

    $$
    \begin{gathered}
    \mathcal{V} =
    \frac{1}{d} \sum_{j=1}^{d}  \max \left(0, \sqrt{C(Z)_{j,j} + \epsilon} \right) + \\
    \frac{1}{d} \sum_{j=1}^{d}  \max \left(0, \sqrt{C(Z^\prime)_{j,j} + \epsilon} \right)
    .\end{gathered}
    $$

* **Covariance**. Forces the features to be uncorrelated:

    $$
    \mathcal{C} =
    \frac{1}{d} \sum_{i,j=1; i\neq j}^{d}  C(Z)_{i,j}^2 +
    \frac{1}{d} \sum_{i,j=1; i\neq j}^{d}  C(Z^\prime)_{i,j}^2
    .$$

The final loss is a weighted sum of the previous terms:

$$
\mathcal{L} = \mathcal{I}  + \lambda \mathcal{V}  + \mu \mathcal{C} 
,$$

where $\lambda$ and $\mu$ are positive hyperparameters. The workings of this
approach are much more intuitive and clear: the regularization terms explicitly
prevent the representation from collapsing, as collapsed representations have
zero feature variance. Decorrelating the features might be useful for
interpretability purposes.


### Applications

We have already talked about the benefits of representation learning in general.
But does contrastive learning have any particular advantages? This is what we
discuss here.

The first advantage is that it provides other training ideas for representation
learning. These techniques have shown great success, reaching state-of-the-art
performance in semi-supervised and transfer learning. The SimCLR model for
images {% cite chen2020SimpleFramework --file 2024-07-16-contrastive-learning %} is an example of this.

A second advantage is the metric representation that they learn. These
representations can later be used in very efficient<d-footnote>There are
techniques to approximately find the closest vectors to the input in a vector
database, such as Locality-Sensitive Hashing or k-d Trees.</d-footnote> vector
searches, allowing us to search for examples in our database that are
**semantically** similar to the input. Thinking of search engines for examples,
we can now find texts that talk about something similar to our search prompt,
without the need to use any common word. With multilingual models, we might
even search semantically in different languages at the same time. We find many
open source language models that have been trained for this semantic encoding,
and that we can readily apply to any search application.

Although not exclusive to contrastive learning, another advantage is that these
techniques allow us to align different encoders in a common representation
space, in particular encoders of different modalities. This is the case of the
CLIP {% cite radford2021LearningTransferable --file 2024-07-16-contrastive-learning %} model we have already
talked about.

Finally, models trained through contrastive learning have zero-shot
classification capabilities. If you have some anchor examples that represent a
category, you can classify a new example by selecting the category with the
closest metric representation. In the case of CLIP, since it has a text
encoder, you can provide any textual label that describes the category, and
apply this inference technique. If there are no textual encoders, you could
still use a small set of labeled data to create a metric representation of
each class. Of course, performance is not as good as with fully-trained models,
specially for very niche categories, but it might still be useful with very few
resources.


## Conclusions

That it is for this post. I hope you found it useful and not too boring! As
always, the best way to learn about anything is to get your hands dirty. In
this case, you can try programming some training loop with a small dataset and
a loss function, such as NT-Xent. You can also try programming some of the
techniques to avoid representational collapse and see how they perform. Don't
doubt in hitting me up with any feedback or corrections!

## References

{% bibliography --file 2024-07-16-contrastive-learning %}