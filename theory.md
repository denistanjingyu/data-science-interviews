# Theoretical interview questions

The list of questions is based on this post: https://hackernoon.com/160-data-science-interview-questions-415s3y2a

Legend: 👶 easy ‍⭐️ medium 🚀 expert

## Supervised machine learning

**What is supervised machine learning? 👶**

A case when we have both features (the matrix X) and the labels (the vector y) 

<br/>

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

Regression is a part of supervised ML. Regression models predict a real number

<br/>

**What is linear regression? When do we use it? 👶**

Answer here

<br/>

**What’s the normal distribution? Why do we care about it? 👶**

Answer here

<br/>

**How do we check if a variable follows the normal distribution? ‍⭐️**

Answer here

<br/>

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

Answer here

<br/>

**What are the methods for solving linear regression do you know? ‍⭐️**

Answer here

<br/>

**What is gradient descent? How does it work? ‍⭐️**

Answer here

<br/>

**What is the normal equation? ‍⭐️**

Answer here

<br/>

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

Answer here

<br/>

**Which metrics for evaluating regression models do you know? 👶**

Answer here

<br/>

**What are MSE and RMSE? 👶**

Answer here

<br/>


## Validation

**What is overfitting? 👶**

When your model perform very well on your training set but can't generalize the test set, because it adjusted a lot to the training set.

<br/>

**How to validate your models? 👶**

Answer here

<br/>

**Why do we need to split our data into three parts: train, validation, and test? 👶**

Answer here

<br/>

**Can you explain how cross-validation works? 👶**

Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting differents training and validation set, in order to reduce the bias that you would have by selecting only one validation set.

<br/>

**What is K-fold cross-validation? 👶**

Answer here

<br/>

**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

Answer here

<br/>


## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Answer here

<br/>

**What is logistic regression? When do we need to use it? 👶**

Answer here

<br/>

**Is logistic regression a linear model? Why? 👶**

Answer here

<br/>

**What is sigmoid? What does it do? 👶**

Answer here

<br/>

**How do we evaluate classification models? 👶**

Answer here

<br/>

**What is accuracy? 👶**

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

<br/>

**Is accuracy always a good metric? 👶**

Accuracy is not a good performance metric when there is imbalance in the dataset. For example, in binary classification with 95% of A class and 5% of B class, prediction accuracy can be 95%. In case of imbalance dataset, we need to choose Precision, recall, or F1 Score depending on the problem we are trying to solve. 

<br/>

**What is the confusion table? What are the cells in this table? 👶**

Confusion table (or confusion matrix) shows how many True positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN) model has made. 

||                |     Actual   |        Actual |
|:---:|   :---:        |     :---:    |:---:          |
||                | Positive (1) | Negative (0)  |
|Predicted|   Positive (1) | TP           | FP            |
|Predicted|   Negative (0) | FN           | TN            |

* True Positives (TP): When the actual class of the observation is 1 (True) and the prediction is 1 (True)
* True Negative (TN): When the actual class of the observation is 0 (False) and the prediction is 0 (False)
* False Positive (FP): When the actual class of the observation is 0 (False) and the prediction is 1 (True)
* False Negative (FN): When the actual class of the observation is 1 (True) and the prediction is 0 (False)

Most of the performance metrics for classification models are based on the values of the confusion matrix. 

<br/>

**What are precision, recall, and F1-score? 👶**

* Precision and recall are classification evaluation metrics:
* P = TP / (TP + FP) and R = TP / (TP + FN).
* Where TP is true positives, FP is false positives and FN is false negatives
* In both cases the score of 1 is the best: we get no false positives or false negatives and only true positives.
* F1 is a combination of both precision and recall in one score:
* F1 = 2 * PR / (P + R). 
* Max F score is 1 and min is 0, with 1 being the best.

<br/>

**Precision-recall trade-off ‍⭐️**

Answer here

<br/>

**What is the ROC curve? When to use it? ‍⭐️**

Answer here

<br/>

**What is AUC (AU ROC)? When to use it? ‍⭐️**

Answer here

<br/>

**How to interpret the AU ROC score? ‍⭐️**

Answer here

<br/>

**What is the PR (precision-recall) curve? ‍⭐️**

Answer here

<br/>

**What is the area under the PR curve? Is it a useful metric? ‍⭐️I**

Answer here

<br/>

**In which cases AU PR is better than AU ROC? ‍⭐️**

Answer here

<br/>

**What do we do with categorical variables? ‍⭐️**

Answer here

<br/>

**Why do we need one-hot encoding? ‍⭐️**

Answer here

<br/>


## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

Answer here

<br/>

**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

Answer here

<br/>

**What is regularization? Why do we need it? 👶**

Answer here

<br/>

**Which regularization techniques do you know? ‍⭐️**

Answer here

<br/>

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

Answer here

<br/>

**How does L2 regularization look like in a linear model? ‍⭐️**

Answer here

<br/>

**How do we select the right regularization parameters? 👶**

Answer here

<br/>

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

Answer here

<br/>

**How L1 regularization looks like in a linear model? ‍⭐️**

Answer here

<br/>

**What’s the difference between L2 and L1 regularization? ‍⭐️**

Answer here

<br/>

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Answer here

<br/>

**What’s the interpretation of the bias term in linear models? ‍⭐️**

Answer here

<br/>

**How do we interpret weights in linear models? ‍⭐️**

If the variables are normalized, we can interpret weights in linear models like the importance of this variable in the predicted result.

<br/>

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

Answer here

<br/>

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

Answer here

<br/>


## Feature selection

**What is feature selection? Why do we need it? 👶**

Answer here

<br/>

**Is feature selection important for linear models? ‍⭐️**

Answer here

<br/>

**Which feature selection techniques do you know? ‍⭐️**

Answer here

<br/>

**Can we use L1 regularization for feature selection? ‍⭐️**

Answer here

<br/>

**Can we use L2 regularization for feature selection? ‍⭐️**

Answer here

<br/>


## Decision trees

**What are the decision trees? 👶**

Answer here

<br/>

**How do we train decision trees? ‍⭐️**

Answer here

<br/>

**What are the main parameters of the decision tree model? 👶**

Answer here

<br/>

**How do we handle categorical variables in decision trees? ‍⭐️**

Answer here

<br/>

**What are the benefits of a single decision tree compared to more complex models? ‍⭐️**

Answer here

<br/>

**How can we know which features are more important for the decision tree model? ‍⭐️**

Answer here

<br/>


## Random forest

**What is random forest? 👶**

Answer here

<br/>

**Why do we need randomization in random forest? ‍⭐️**

Answer here

<br/>

**What are the main parameters of the random forest model? ‍⭐️**

Answer here

<br/>

**How do we select the depth of the trees in random forest? ‍⭐️**

Answer here

<br/>

**How do we know how many trees we need in random forest? ‍⭐️**

Answer here

<br/>

**Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️**

Answer here

<br/>

**What are the potential problems with many large trees? ‍⭐️**

Answer here

<br/>

**What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀**

Answer here

<br/>

**What happens when we have correlated features in our data? ‍⭐️**

Correlated features in general don't improve models (although it depends on the specifics of the problem like the number of variables and the degree of correlation), but they affect specific models in different ways and to varying extents:

- For linear models (e.g., linear regression or logistic regression), multicolinearity can yield solutions that are wildly varying and possibly numerically unstable.

- Random forests can be good at detecting interactions between different features, but highly correlated features can mask these interactions.

More generally, this can be viewed as a special case of Occam's razor. A simpler model is preferable, and, in some sense, a model with fewer features is simpler. 
<br/>


## Gradient boosting

**What is gradient boosting trees? ‍⭐️**

Answer here

<br/>

**What’s the difference between random forest and gradient boosting? ‍⭐️**

- Boosting is based on weak learners (high bias, low variance). In terms of decision trees, weak learners are shallow trees, sometimes even as small as decision stumps (trees with two leaves). Boosting reduces error mainly by reducing bias (and also to some extent variance, by aggregating the output from many models).
- Random Forest uses fully grown decision trees (low bias, high variance). It tackles the error reduction task in the opposite way: by reducing variance. The trees are made uncorrelated to maximize the decrease in variance, but the algorithm cannot reduce bias (which is slightly higher than the bias of an individual tree in the forest). Hence the need for large, unpruned trees, so that the bias is initially as low as possible.
- Boosting is sequential, RF grows trees in parallel. 

<br/>

**Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️**

Answer here

<br/>

**Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️**

Answer here

<br/>

**Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀**

Answer here

<br/>

**What are the main parameters in the gradient boosting model? ‍⭐️**

Answer here

<br/>

**How do you approach tuning parameters in XGBoost or LightGBM? 🚀**

- Choose a relatively high learning rate. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems. Determine the optimum number of trees for this learning rate. XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus returns the optimum number of trees required.
- Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
- Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
- Lower the learning rate and decide the optimal parameters .

<br/>

**How do you select the number of trees in the gradient boosting model? ‍⭐️**

Answer here

<br/>



## Parameter tuning

**Which parameter tuning strategies (in general) do you know? ‍⭐️**

Answer here

<br/>

**What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️**

- In Grid Search, we try every combination of a preset list of values of the hyper-parameters and evaluate the model for each combination. The pattern followed here is similar to the grid, where all the values are placed in the form of a matrix. Each set of parameters is taken into consideration and the accuracy is noted. Once all the combinations are evaluated, the model with the set of parameters which give the top accuracy is considered to be the best.
- Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It tries random combinations of a range of values. To optimise with random search, the function is evaluated at some number of random configurations in the parameter space.
- Random search works best for lower dimensional data since the time taken to find the right set is less with less number of iterations.
- The best strategy for your problem is the one that finds the best value the fastest and with the fewest function evaluations and it may vary from problem to problem. While less common in machine learning practice than grid search, random search has been shown to find equal or better values than grid search within fewer function evaluations for certain types of problems.

<br/>


## Neural networks

**What kind of problems neural nets can solve? 👶**

Answer here

<br/>

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

Answer here

<br/>

**Why do we need activation functions? 👶**

The purpose of an activation function is to add some kind of non-linear property to the function, which is a neural network. Without the activation functions, the neural network could perform only linear mappings from inputs x to the outputs y.

<br/>

**What are the problems with sigmoid as an activation function? ‍⭐️**

The sigmoid function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.

<br/>

**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

Answer here

<br/>

**How we can initialize the weights of a neural network? ‍⭐️**

Answer here

<br/>

**What if we set all the weights of a neural network to 0? ‍⭐️**
First, neural networks tend to get stuck in local minima, so it's a good idea to give them many different starting values. You can't do that if they all start at zero.

Second, if the neurons start with the same weights, then all the neurons will follow the same gradient, and will always end up doing the same thing as one another.

<br/>

**What regularization techniques for neural nets do you know? ‍⭐️**

1) Activity Regularization: Penalize the model during training base on the magnitude of the activations.
2) Weight Constraint: Constrain the magnitude of weights to be within a range or below a limit.
3) Dropout: Probabilistically remove inputs during training.
4) Noise: Add statistical noise to inputs during training.
5) Early Stopping: Monitor model performance on a validation set and stop training when performance degrades.

<br/>

**What is dropout? Why is it useful? How does it work? ‍⭐️**

Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

You can imagine that if neurons are randomly dropped out of the network during training, that other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.

The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.

<br/>


## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

Back-propagation is the essence of neural net training. It is the practice of fine-tuning the weights of a neural net based on the error rate (i.e. loss) obtained in the previous epoch (i.e. iteration). Proper tuning of the weights ensures lower error rates, making the model reliable by increasing its generalization.

<br/>

**Which optimization techniques for training neural nets do you know? ‍⭐️**

Answer here

<br/>

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

Answer here

<br/>

**What’s the learning rate? 👶**

Answer here

<br/>

**What happens when the learning rate is too large? Too small? 👶**

Answer here

<br/>

**How to set the learning rate? ‍⭐️**

Answer here

<br/>

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

Answer here

<br/>

**When would you use Adam and when SGD? ‍⭐️**

Answer here

<br/>

**Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️**

Answer here

<br/>

**How do we decide when to stop training a neural net? 👶**

Answer here

<br/>

**What is model checkpointing? ‍⭐️**

Answer here

<br/>

**Can you tell us how you approach the model training process? ‍⭐️**

Answer here

<br/>


## Neural networks for computer vision

**How we can use neural nets for computer vision? ‍⭐️**

Answer here

<br/>

**What’s a convolutional layer? ‍⭐️**

Answer here

<br/>

**Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️**

Answer here

<br/>

**What’s pooling in CNN? Why do we need it? ‍⭐️**

Answer here

<br/>

**How does max pooling work? Are there other pooling techniques? ‍⭐️**

Answer here

<br/>

**Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀**

Answer here

<br/>

**What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️**

Answer here

<br/>

**What kind of CNN architectures for classification do you know? 🚀**

Answer here

<br/>

**What is transfer learning? How does it work? ‍⭐️**

Answer here

<br/>

**What is object detection? Do you know any architectures for that? 🚀**

Answer here

<br/>

**What is object segmentation? Do you know any architectures for that? 🚀**

Answer here

<br/>


## Text classification

**How can we use machine learning for text classification? ‍⭐️**

Answer here

<br/>

**What is bag of words? How we can use it for text classification? ‍⭐️**

Answer here

<br/>

**What are the advantages and disadvantages of bag of words? ‍⭐️**

Answer here

<br/>

**What are N-grams? How can we use them? ‍⭐️**

Answer here

<br/>

**How large should be N for our bag of words when using N-grams? ‍⭐️**

Answer here

<br/>

**What is TF-IDF? How is it useful for text classification? ‍⭐️**

Answer here

<br/>

**Which model would you use for text classification with bag of words features? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️**

Answer here

<br/>

**What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️**

Answer here

<br/>

**Do you know any other ways to get word embeddings? 🚀**

Answer here

<br/>

**If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️**

Answer here

<br/>

**How can you use neural nets for text classification? 🚀**

Answer here

<br/>

**How can we use CNN for text classification? 🚀**

Answer here

<br/>


## Clustering

**What is unsupervised learning? 👶**

Answer here

<br/>

**What is clustering? When do we need it? 👶**

Answer here

<br/>

**Do you know how K-means works? ‍⭐️**

Answer here

<br/>

**How to select K for K-means? ‍⭐️**

Answer here

<br/>

**What are the other clustering algorithms do you know? ‍⭐️**

Answer here

<br/>

**Do you know how DBScan works? ‍⭐️**

Answer here

<br/>

**When would you choose K-means and when DBScan? ‍⭐️**

Answer here

<br/>


## Dimensionality reduction
**What is the curse of dimensionality? Why do we care about it? ‍⭐️**

- The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.
- Working with data becomes more computationally demanding as the number of dimensions increases.
- Highly correlated dimensions can harmfully impact other statistical techniques which rely upon assumptions of independence. This could lead to much-dreaded problems such as over-fitting.
- In higher dimensions our data are more sparse and more similarly spaced apart. This makes most distance functions less effective.

<br/>

**Do you know any dimensionality reduction techniques? ‍⭐️**

Ratio of missing values
Low variance in the column values
High correlation between two columns
Principal component analysis (PCA)
Candidates and split columns in a random forest
Backward feature elimination
Forward feature construction
Linear discriminant analysis (LDA)
Neural autoencoder
t-distributed stochastic neighbor embedding (t-SNE)

<br/>

**What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️**

Answer here

<br/>


## Ranking and search

**What is the ranking problem? Which models can you use to solve them? ‍⭐️**

Answer here

<br/>

**What are good unsupervised baselines for text information retrieval? ‍⭐️**

Answer here

<br/>

**How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️**

Answer here

<br/>

**What is precision and recall at k? ‍⭐️**

Answer here

<br/>

**What is mean average precision at k? ‍⭐️**

Answer here

<br/>

**How can we use machine learning for search? ‍⭐️**

Answer here

<br/>

**How can we get training data for our ranking algorithms? ‍⭐️**

Answer here

<br/>

**Can we formulate the search problem as a classification problem? How? ‍⭐️**

Answer here

<br/>

**How can we use clicks data as the training data for ranking algorithms? 🚀**

Answer here

<br/>

**Do you know how to use gradient boosting trees for ranking? 🚀**

Answer here

<br/>

**How do you do an online evaluation of a new ranking algorithm? ‍⭐️**

Answer here

<br/>


## Recommender systems

**What is a recommender system? 👶**

A recommender system, or a recommendation system, is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications.

<br/>

**What are good baselines when building a recommender system? ‍⭐️**

Answer here

<br/>

**What is collaborative filtering? ‍⭐️**

Answer here

<br/>

**How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️**

Answer here

<br/>

**What is the cold start problem? ‍⭐️**

Personalized recommender systems take advantage of users past history to make predictions. The cold start problem concerns the personalized recommendations for users with no or few past history (new users). Providing recommendations to users with small past history becomes a difficult problem for CF models because their learning and predictive ability is limited. 

<br/>

**Possible approaches to solving the cold start problem? ‍⭐️🚀**

Use auxiliary information (multimodal information, side information, etc.) to overcome the cold start problem

<br/>


## Time series

**What is a time series? 👶**

A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data.

<br/>

**How is time series different from the usual regression problem? 👶**

The biggest difference is that time series regression accounts for the autocorrelation between time events, which always exists, while in normal regression, independence of serial errors are presumed, or at least minimized.

<br/>

**Which models do you know for solving time series problems? ‍⭐️**

Autoregression (AR)
Moving Average (MA)
Autoregressive Moving Average (ARMA)
Autoregressive Integrated Moving Average (ARIMA)
Seasonal Autoregressive Integrated Moving-Average (SARIMA)
Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
Vector Autoregression (VAR)
Vector Autoregression Moving-Average (VARMA)
Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
Simple Exponential Smoothing (SES)
Holt Winter’s Exponential Smoothing (HWES)

<br/>

**If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️**

Answer here

<br/>

**You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**What are the problems with using trees for solving time series problems? ‍⭐️**

Random Forests don’t fit very well for increasing or decreasing trends which are usually encountered when dealing with time-series analysis, such as seasonality.
<br/>


