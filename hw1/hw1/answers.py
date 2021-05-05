r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False. It allows estimation of the performance of the model on new data, so we can estimate how well it generalized. 
2. False. We should pick the best subset of the data that can maximize the representation of our data set. A random pick can give us bad representation
   and therefore give us a bad results. 
3. True. The test set should be used when evaluating the performance of the model and nothing else. 
4. True


"""

part1_q2 = r"""
**Your answer:**
Yes, we need to save data section for validation for hyperparameters tuning and  also to choose the best model. Using the test set for that purpose means overfitting on the test set. In such case we can't evaluate how well the model generalizes.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Increasing k leads to improved generalization for unseen data up to a certain point. 
Large values of k will determine the class of the unseen data according to most of the examples in the dataset in high values, 
which will be incorrect classification if the correct class is smaller.On the other hand, a value of k that is too small might
also be incorrect if the closest example belongs to the incorrect class and that can happen if there are
small amount of examples.Therefore the ideal value of k should be neither too small nor too large.
"""

part2_q2 = r"""
**Your answer:**
1. Training on the entire train-set with various models and then selecting the best model with respect
to train-set accuracy is bad practice, since it leads to overfitting on the training data. We will then select
a model that performs best on the training data, while it may very well be very wrong on new unseen data.

2. Training on the entire train-set with various models and then selecting the best model with respect
to test-set accuracy is somewhat better than (1), since we are determining the selected model according to
untouched test-set. However, the selected model is very much influenced by the selected test set.
Dividing the data differently might lead to completely different and inconsistent results.
Using K-fold CV solves both problems. Overfitting is reduced a lot with K-fold CV since the selected model
is determined by an average of the accuracy on different train-sets (each time a new train-set is selected),
and the model is not biased towards one selected train-set.
In addition, the selected model is not influenced by one test-set, since each time the model is being tested
on a different test-set, which leads to consistent results and lack of bias.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
$\Delta > 0$ is a hyperparameter of the model, which allows some error for wrong predictions, the margin. We also added
regularization on the weights, and the weights can be thought of as the distance between the sample and the decision 
boundary. Changing $\Delta$ will result in a different model with different weights, so the value of $\Delta > 0$ is
arbitrary.

"""

part3_q2 = r"""
**Your answer:**
1. The linear model learns the locations of each label and then fits the digit which resembles the current sample 
 the most. For example, in the first error the model classified a 5 digit as a 6 digit. If we look closely in the
 written digit, it is not a 'typical' 5 digit- it resembles the shape of a 5 digit, and it is quite clear why the model
 wrongly classified this specific sample.
 
2. It is similar to KNN because with KNN we can see that examples that are closer to each other in their features
tend to be represented with certain values (colors). Same as here, where a new unseen digit (sample) that is
close to a certain weight image in the colors - is similar to it in the features, and therefore will be given
the same classification.

"""

part3_q3 = r"""
**Your answer:**
1. The learning rate we chose is possibly too high because the model converges fast and has a sudden big error after 
the convergence. This can be explained as a big step taken which is overshooting the minimum. A learning rate too low
would cause a slow convergence and would reach lower accuracy in the same number of epochs.

2. The model is slightly overfitting the train set sinch there is a ~7% difference in the accuracies of the train and 
the test set. It reaches ~88% accuracy on the test set, so it is not highly overfitted.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
In a residual plot, the ideal pattern would be that the entire data set points lie on the line that represents y - y_hat = 0.
This way the prediction for the entire data set is correct and the error rate will be zero.
If we look at the residual plot in the last part, we can see that the fitness level of the trained model is pretty high, and it describes
the real world pretty well given the fact that most of the test data points lie close to the line y - y_hat = 0. 
It means we were able to generalize our model pretty well based on the training data. Additionally, looking at the plot for the top-5 features and looking at the final plot after CV, we can identify that the former is less accurate, since more data points lie further from the correct prediction line, while the latter indicates choosing the best hyperparameters and thus improving the model,and therefore shows an improvement when more data points lie closer to the correct prediction line.

"""

part4_q2 = r"""
**Your answer:**
1. Yes, as the linearity is considered relatively to $\mat{W}$ and not relatively to the original features. 
2. Yes, as any known non-linear relation between the original features can be obtained by feature engineering at the price of higher dimension.
3. The decision boundary after adding the non-linear features will be of higher space. It would still be a hyperplane as the model of linear regression always learn a linear separator (as it is still linear in terms of $\mat{W}$).
"""

part4_q3 = r"""
**Your answer:**
1. We used ```np.logspace``` instead of ```np.linspace``` as logarithmic scale enables us to search a bigger space with dramatically different values quickly.
In addition, it enables us to be more sensitive regard smaller values as we expect less variation in the results between bigger values of the regularization coefficient.

2. By performing K-fold CV, we are fitting the data K times, since we are dividing the data into K different folds while in each of the K iterations we are fitting (K-1) parts of the data and predicting on the remaining part. If we look at the code above, and including hyperparameters examination,
we have 3 options for the degree and 20 options for lambda, which make 60 different options for parameters combinations. In addition, we are using 3-fold CV, and so overall we are fitting 3 times on each of the parameters combination, to make 180 fittings in total. 
"""

# ==============
