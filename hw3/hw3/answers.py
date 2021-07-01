r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 512
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.5
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "SCENE I"
    # ========================
    return start_seq, temperature


part1_q1 = r"""

We split the corpus into sequences instead of training on the entire text because texts in their
nature consist of words which are short character sequences, and therefore in real world (also done
by humans) in order to be able to predict the next character or few characters we want to train by looking
at a small text window rather than the entire corpus in whole. Note that a small text window
could also be of size 1, which means we train on one character in each iteration.
Additionally, we want to conclude relations between characters and between words and characters,
in order to predict the next most probable character, which will be harder to do if we look at the
entire corpus dataset as a whole.

"""

part1_q2 = r"""

The generated text clearly shows memory longer than the sequence length because the network was added
a memory component, which can remember inputs and then recall those inputs at later time stamps.
It looks also at states prior to the last one and not just the last one, hence preserving "memory"
which can be longer than the sequence length.
Explanation: much like LSTM's, the GRU's have a memory cell, and they solved the vanishing gradients
problem by introducing a new memory gate, that allowed hidden state to either pass through time (remembered)
or not (forgot). This enabled the modeling of sequence with much greater length than before.

"""

part1_q3 = r"""

Unlike previous tasks that we dealt with, in which the different batches did not have a specific order
to them, here the batches do have a specific order: we are analyzing text bits of a larger corpus, where
the learning order is important: we would like to learn the text in the order it appears in the corpus,
because eventually we would like to generate new characters and words. If we shuffle the order of batches,
we will 'confuse' the model to learn unrelated and unordered bits of text, which in turn will generate
new text in an unrelated and unordered manner, which is not what we desire.

"""

part1_q4 = r"""

We will answer all 3 sub-questions together, referring to temperature as 'temp.' or 'T':
The temp. hyperparameter of networks is used in order to control the randomness of predictions by scaling
the logits before applying softmax. It increases the sensititvity to low probability candidates.
For $T = 1$ which is normally the default value, no scaling is being done and we get the value as is.
Using T that is lower than 1, for instance $T = 0.5$, the model computes the softmax on the scaled logits,
i.e in this case $logits/0.5$, which increases the logits values.
Next, performing softmax on larger values makes the LSTM more confident but more conservative in its samples.
'More confident' means less input is needed to activate the output layer, and 'more conservative' means
it is less likely to sample form unlikely candidates.
On the other hand, using T that is higher than 1, for instance $T = 1.5$, we get a softer probability
distribution over the classes, which in turn makes the RNN more diverse and more error-prone, meaning it
is more likely to pick lower probability candidates and makes more mistakes.
Lastly, the softmax function normalizes the candidates at each iteration by ensuring the network outputs
are all between 0 and 1.
We would like to be more confident in order to be less mistaken (even if it means we are more conservative),
because what matters most to us is the accuracy of the results, and therefore as explained earlier,
we are going to use temp. values that are lower than 1 rather than higher than 1

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 512
    hypers['z_dim'] = 256
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 1e-4
    hypers['betas'] = (0.9, 0.99)  
    # ========================
    return hypers


part2_q1 = r"""

The $\sigma^2$ in the VAE model represents the variance in the normal distribution.
Generally, if $\sigma^2$ will have too big of a value, we will have to take a very large number of samples to find
out the true expectation. When the variance is so high, using it to learn different models becomes hard.
In good practices using VAE, we prefer a low-variance gradient estimator based on the reparametrization trick.
And so in general, in different examples we've seen, high variance tends to be worse and low variance tends to
be better, so we need to lower the value, but making sure it is not becoming too low.
"""

part2_q2 = r"""

1.  In the first place, we want encodings that are as close as possible to each other while maintaining being distinct
    from each other. This allows smooth interpolation and the generation of samples.
    The KL divergence loss's purpose is to ensure this, while the divergence value between two probability
    distributions measures how much they are different from each other.
    By minimizing this loss we are optimizing the prob. dist. parameters (among them - $\sigma$) to be as close as it
    can to the target dist.
    Regarding the reconstruction loss, it is usually either the MSE or CE loss between the output and the input.
    Its purpose is to penalize the network for creating outputs that are different from the input. 

2.  The KL divergence encourages the encoder to distribute all input encodings in an even manner around the center
    of the latent space. If it clusters them apart into specific regions and away from the center, it will get a penalty.

3.  The benefit of all the input encodings being evenly distributed around the center of the latent space is that unlike
    when the encodings are densely placed randomly near the center of the latent space, where the decoder finds it
    very hard to decode anything meaningful from the space, when the two losses are optimized together we maintain the
    similarity of nearby encodings locally, and globally it is densely packed near the center of the latent space.
    And then when we reach the equilibrium of the two losses, then when randomly generating a sample (sampling a vector from
    the same prior dist.), the decoder can successfully decode it, with no gaps between clusters and a smooth mix of features. 
    
"""

part2_q3 = r"""

Maximizing the evidence distribution is crucial for making the output sample as close to the dataset as we can get.
Maximizing it will result in decoding output to be as close to the dataset with high probability.
So our expectation is that our whole encoding - decoding system will work better if we maximize the evidence distribution.
"""

part2_q4 = r"""

Modeling the log of the latent-space variance instead of using the variance has the following pros:

1.  $\bb{\sigma}^2_{\bb{\alpha}}$ is a positive value by definition and the $log(\bb{\sigma}^2)$ will output positive value.

2.  We get numerical stability because we map small values in the interval [0,1] to (-inf, log(1)] so we get better numerical stability due to that mapping.
    Furthermore $\bb{\sigma}^2_{\bb{\alpha}}$ is a relatively small value 1>>$\bb{\sigma}^2_{\bb{\alpha}}$>0.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 4
    hypers['label_noise'] = 0.2
    hypers['data_label'] = 1
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.99)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.99)
    # ========================
    return hypers


part3_q1 = r"""

The GAN as we saw consists out of 2 parts - discriminator and generator. 
During the training process of both of them we generate fake data using the generator and feed it
into the discriminator along with real data in order to discriminate between them.
The discriminator simply behaves as a classifier and during that phase we keep the generator constant (we don't keep the 
generator gradients) in order to "help" the discriminator to converge and to allow it to learn the generator flaws.
During the generator training phase we need to keep of course its gradients in each step, but also we keep the
discriminator gradients as we want to backpropagate all the way from very end of the entire model, which is the 
discriminator output.  
"""

part3_q2 = r"""

1.  No, we shouldn't decide stopping training solely based on the generator loss as the it's score depends on the 
    discriminator performance: If the discriminator isn't accurate enough and the generator loss is very low it simply 
    means that the generator is performing well in fooling the discriminator. But since, the discriminator is doing a 
    lousy job in the first place, it doesn't mean that in overall the entire model is performing good enough.
    
2.  This means that the discriminator is ahead of the generator training wise. In other words it tells us that the 
    discriminator can easily tell the difference between the real and the generated data. While that, the generator 
    is trying to "catch" the discriminator and is learning from the results it provides.
    
"""

part3_q3 = r"""

First, while the VAE and the GAN are similiar in their abilities to create new generated data, similiar to existing 
real one, they are very different in the approach they both take to achieve that goal. While the VAE learning approach 
is to compress the data correctly in order to be able to reconstruct it later, meaning it focuses directly on the data.
The GAN approach is to train a "cop" in a way so it will be able to distinguish between real and fake data, along 
with training a "counterfeiter" so it will be able to fool the cop and vice versa. Meaning, its approach is to create
a competition between the two adversaries so the focusing on the data is indirect in some sense.
We can see that in the VAE we got slightly better results due to the reasons above. Also we can see that because the VAE
focuses directly on the pictures, it also learned to distinguish meaningful areas such as the face part and to ignore
the background parts. Thats why the background parts are more blurry in the VAE and the face is much more precise, while
in the GAN there is much more similiarity (for better, but also mostly for worse) in the different parts of the generated
pictures.

Another possible reason for the poorer results in the GAN (compared to the VAE) since its training process is hard.
The generator and the discriminator are constantly trying to improve on the cost of each other so we can think at it
as trying to shoot at a moving target rather then the typical and constant training process of models we learned so far, including 
the VAE among them. 
"""


# ==============
