r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. The Jacobian's tensor shape of the output of the layer w.r.t the input X is of shape (batch, ins, batch, outs). Therefore it's $(N, \textit{in_features}, N, \textit{out_features}) = (128,1024,128,2048)$.
    
    
2. In order to store the Jacobian in memory of GPU or RAM we will need 128GB. Assuming we are using single-precision floating point (32 bits) to represent our tensors:
    we have 128 batches with 128 samples each, and $\textit{in_features} = 1024$, $\textit{out_features} = 2048$. The computation is:
    $(\textit{num of samples}) \cdot  (\textit{num of batches}) \cdot (features_{in}) \cdot (features_{out}) \cdot 4(\textit{bytes})
    = 4 \cdot 128 \cdot 128 \cdot 1024 \cdot 2048 = 2^2 \cdot 2^7 \cdot 2^7 \cdot 2^{10} \cdot 2^{11} = 2^{37}
    = 2^7 \cdot 2^{30} = 128 \cdot 2^{30} = ~ 128GB$

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.03
    reg = 0.01
    lr_momentum = 3e-3
    lr_rmsprop = 15e-5
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. In the graph, we can see that the accuracy with $dropout = 0.4$ is greater than $dropout = 0$, and $dropout = 0$ is greater than $dropout = 0.8$.
The results are matching our expectations since some dropout improves the generalization of the model (i.e less likely to overfit).
In our case, the training set is relatively small in comparison to the test set, thus more likely to overfit. Therefore we can see
in the results that no dropout produce lesser results than some dropout ($dropout=0.4$).

2. We can see that low-dropout settings produce much higher results than high-dropout settings. In fact, the high-dropout setting
shows lesser results than no-dropout settings. The explanation for this is that too much dropout (in our case we're dropping 80% of the neurons),
hinders the learning process.

"""

part2_q2 = r"""
Cross-Entropy Loss is is known for "punishing" over wrong predictions. Thus, wrong prediction have larger effect over the loss than
correct predictions. In a case which there are wrong predictions over one class, and overall correct predictions
in the rest of the classes, it is possible for the test loss to increase while the accuracy also increases.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. Regular block has 2 convolution layers, each has 3X3 kernel and 256 channels as input and output therefore it's parameters count is: $2\cdot((3\cdot3\cdot256 + 1)\cdot256) = 1,180,160$.  
    Bottleneck block has 3 convolution layers:
        a. 1X1 256 -> 64
        b. 3X3 64 -> 64
        c. 1X1 64 -> 256
    Therefore it's parameters count is $(256+1)\cdot64 + (3\cdot3\cdot64 + 1)\cdot64 + (64+1)\cdot256 = 70,016$.
    We can see that the bottleneck has less parameters than the regular block thus it needs much less resources as memory and computing power in order to computre.
    
2. Lets assume input size is (256, H, W) and that the following are one floating point operation:
    - basic math operations
    - Relu (per element)
    The formula is $K \cdot K \cdot input \cdot H_{out} \cdot W_{out} \cdot output$ for a KxK input -> output layer forward pass.
    The regular block:
    First layer = $3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256$
    2 X Relu =  $2 \cdot 256 \cdot H \cdot W $
    Second layer = $ 3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256 $
    Residual connection = $ 256 \cdot H \cdot W $
    All of those sum up to $ 1,180,416 \cdot H \cdot W$
    
    The bottleneck block:
    First layer = $1 \cdot 1 \cdot 256 \cdot H \cdot W \cdot 64 $
    2 X Relu (64) = $2 \cdot  64 \cdot H \cdot W $
    Second layer = $3 \cdot 3 \cdot 64 \cdot H \cdot W \cdot 64 $ 
    Third layer = $1 \cdot 1 \cdot 64 \cdot H \cdot W \cdot 256 $
    Residual connection = $256 \cdot H \cdot W $
    Relu (256) = $256 \cdot H \cdot W $
    All of those sum up to $70,272 \cdot H \cdot W$
    
   In conclusion from those calculations we can see that the bottleneck block is much better in terms of computing.
    
3. The regular block mostly effects the input spatially - we can conclude it because it maintains the number of feature maps.
   The bottleneck block effects both spatial and feature map of the input - we can conclude it because the first convolution layer 1X1 maintaining the spatiallity of the input
   and map the input, afterwards the input goes through a similar convolutions as the regular block that operates on the spatiallity of the input.
   Therefore we conclude that the bottleneck block effects mostly on the spatial aspect but also on the feature map.

"""

part3_q2 = r"""
1. According to our results from the experiments, in general the deeper the network is the better(higher) the accuracy is,
unless it reaches a threshold in which the more deep it gets the worsethe results of accuracy are. 
We saw that we got the ***best results for $L=4$***, whereas for both $L=2$and $L=16$ we got lower accuracy. We think this is the case because in general,
the deeper the network the more complex problems it can learn and generalize, and thus the accuracy grows. 
However too deepof a network might badly affect the accuracy since the number of parameters will grow and it will cause the network to be somewhat instable and thus to poorer accuracy results. 

2. For $L=8$ we realized that the network was not trainable, both for $K=32$ and $K=64$. When we tried to do that we got a runtime "division by 0" error,
which probably occurred due to some values being so small that they were considered as 0 in the program, 
and then when the network computed parameters and did calculations it ended up dividing by 0. To partially resolve it we could make sure that for all of the parameters in the network,
if they go below some threshold we will manually multiply them by a certain constant. Another solution could be to make sure beforehand that the network is not going to work with
parameters that are too small in the beginning, where there will be a chance that they will decrease and will be considered to be 0.

"""

part3_q3 = r"""
In experiment 1.2 we changed the number of filters per layer (K). What we noticed is that the more K grew the more we observed that the test loss is increasing with epochs. 
This is more visible with $L=4$ for example than with $L=2$. Otherwise with train and test accuracy, it is hard to pinpoint a trend in the graphs for all L's that were examined.
It may be the case that the more filters we use in a network the more it is hard for it to produce good learning results, which in turn causes the loss to increase.
Comparing it to the results of experiment 1.1 we can see that it pretty much matches the results in that in 1.1 we also saw this trend with test loss when K increases.

"""

part3_q4 = r"""
In this experiment we checked different values of L with a constant K which was $K=[64, 128, 256]$. What we found out was that the more L grew in value, the more the test loss trend decreased.
This matches our results and explanations from experiment 1.1 where ***up to a certain point***, the deeper the network the better the results are (lower loss and higher accuracy). 
Same happens here where the test loss decreased from $L=1$ to $L=3$.Less clear trends are that the more L grew in value, the less the train accuracy increased and the more
the test accuracy increased, but this may also be a statistical (measurement) error.

"""

part3_q5 = r"""
In this experiment we introduced the ResNet (as opposed to the ConvClassifier) for experiments 1.1 and 1.3. We increased L with a constant K of 32 and then we increased the L with constant K of [64, 128, 256].
The results are:
- For the $K=32$, the bigger the L the lower the accuracy for both train and test,looking at L values of 8, 16, 32.
  We already saw this trend with experiment 1.1, where starting from a certain L value, the bigger the L value is the lower the accuracy is, and this L may very well be $L=8$
  (in experiment 1.1 we checked with L=2 to L=16 and we noticed a decrease with L=16 (which matches the results here). The explanation can be similar to that of experiment 1.1, even with ResNet.
- For the $K=[64, 128, 256]$, the bigger the L the lower the accuracy on both train and test. This opposes the results of experiment 1.3, where there (this time with $K=[64, 128, 256]$) we experienced
  a decrease in test loss (which will cause better accuracy) the more L grew. This can be explained by the fact that in experiment 1.4 we are using ResNet, which essentially uses Residual Blocks thus skipping
  certain convolutional layers. It turns out that by doing so the accuracy decreases the more L grows, perhaps because of the fact that with residual blocks the network is skipping crucial parts the more
  layers are added, whereas with a traditional convolutional network the more layers are added without skipping the more complex the network is and the more complex parameters it can learn, which results
  in better accuracy overall.

"""

part3_q6 = r"""
1. In this section we implemented a convolutional neural network and then tried different hyper-parameters to see which one will give us the best result. We implemented two parts the same as the ConvClassifier
that was implemented earlier: One part is a convolutional layers part, with 5 conv layers and batchnorm and ReLU layers after each conv layer, and maxpooling layer after each layer except for the last one.
The second part consists of 6 linear layers, which end up with num_classes. In our experiments we tried adding dropout of 0.4 (which happened to be a good results in notebook 2 results),
as well as $stride=2$ and $dilation=2$ to the conv layers. It ended up with bad results. We then removed dilation and left only the stride and got about 36% test accuracy. We then removed the
stride as well and improved the test accuracy to 58%. We then removed the dropout too and got ~88% of train accuracy and ~76% of test accuracy. We then tried 6 different values of learning rates, ranging from
0.0001 to 0.1 and found that the best result is with the default one which was $LR=0.001$. We then tried 7 different values of regularization terms, ranging from 0.0001 to 0.3 and got the best results
again for $REG=0.001$ with ~88% of train accuracy and ~77% of test accuracy. We then tried to increase the number of epochs, to 15, 20 and 25, and ended up having little lower
results for test accuracy and much higher results for train accuracy, reaching 93, 95 and 97%, which we conclude that is overfitting on the training set. Therefore we conclude that $EPOCHS=10$ is a good
value for this hyper-parameter. And so that gave us our final network architecture.

2. In the results we can see that the test loss is very noisy, although decreasing the more L grows. Additionally the train accuracy decreases lightly with the increase of L, and same with the test accuracy.
This supports the results of experiment 1.3 where the more L grew the more the test loss decreased, and somewhat contradicts the results of experiment 1.4. This can result from the fact that out implemented
network is a regular convolutional network the same as in experiment 1.3, and not a Residual blocks-based network as in experiment 1.4, and therefore the results are similar to the results of experiment 1.3
and are different from those of experiment 1.4, and for the reasons mentioned in the above sections.

"""
# ==============
