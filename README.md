# Machine Learning Algorithms

Implementations of commonly-used machine learning algorithms from scratch using only the [numpy](https://numpy.org/) library. Each algorithm is standalone with no other dependencies of other algorithms. 

Each models is intended for a transparent look into their implementation. They are not intended to be efficient or used to practical applications, but simply offer aid to anyone studying machine learning.


## User Guide

All implementations are based on the [scikit-learn](https://scikit-learn.org/) and [Keras](https://keras.io/) libraries and follow a similar class style and structure.

### Model Creation
All models are created by initializing a class with their hyperparameters. Each model already has default hyperparameters so models can be created without any parameters as well.

```python
classifier = LogisticRegression(penalty='l1', C=0.001) # specify hyperparameters
regressor = RidgeRegression() # use default hyperparameters
```

Since various other parameters are setup when creating a model. It is recommended to completely reinitialize the model rather than changing the hyperparameter.

```python
tree_clf = DecisionTreeClassifier(criterion='gini')
tree_clf.criterion = 'entropy' # will not work, do not use
tree_clf = DecisionTreeClassifier(criterion='entropy') # use this instead
```

### Model Training

All models are trained using the `fit(X, y)` method. This will always take parameters `X`, a square matrix of training features, and `y`, the training labels. If the algorithm uses gradient descent, it may also take optional parameters for the `epochs` and `lr` to override the defaults. If the model uses gradient descent, the `fit(X, y)` method will return two lists for the training loss and evaluation metric (accuracy or R2 score) per training epoch. Otherwise, the model will return the final loss and evaluation metric from the trained model, equivalent to calling `evaluate(X, y)`.

```python
training_loss, training_acc = classifier.fit(X, y) # returns Tuple[list, list]
loss, r2 = regressor.fit(X, y) # returns Tuple[float, float]
```

### Model prediction

All models have a `predict(X)` method which can be called after training. This will return the predicted values based on the weights learned from training.

```python
y_pred = classifier.predict(X) # returns class labels
y_pred = regressor.predict(X) # returns real value predictions
```

In addition, some classifiers have a `predict_proba(X)` and `predict_log_proba(X)` to get the class probabilities and log probabilities.

```python
y_pred_prob = classifer.predict_proba(X)
y_pred_log_prob = classifer.predict_log_proba(X)
```

### Model evaluation

To evaluate a model, it is recommended to run `predict(X)` and use your evaluation metrics of choice (accuracy, R2 score, F1 score, cross entropy, MSE, etc.). However, in order to get a quick and rough estimate of the model performance, all models have an `evaluate(X, y)` method which will return the default loss and evaluation metric. These metrics are model specific.

```python
ce, acc = classifier.evaluate(X, y) # cross entropy and binary accuracy
mse, r2 = regressor.evaluate(X, y) # mean square error and R2 score
```


## Algorithms Available

Various algorithms for supervised learning have been implemented. Each are separated into their own category located in their respective subfolder. All implementations are completely standalone so there are no other dependencies and the class can be used immediately out of the box.

### Linear Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Logistic Regression
	* L1 Penalty
	* L2 Penalty
	* ElasticNet Penalty

### Decision Trees

- Decision Tree Classifier
	* Gini Split
	* Entropy Split
	* Misclassification Split
- Decision Tree Regressor
	* Mean Squared Error Split
	* Mean Absolute Error Split
	* Poisson Deviance Split

### Support Vector Machines

- Support Vector Classifier (in-progress)
- Support Vector Regressor (in-progress)

### Neural Networks

- Multilayer Perceptron Regressor
- Multilayer Perceptron Classifier
