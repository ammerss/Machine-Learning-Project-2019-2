# Machine Learning Project 2019-2

<details>
<summary>Assignment 01</summary>
  
# Assignment 01

## General Instruction

#### Jupyter Notebook

```console
- Write programming codes in python
- Use Jupyter Notebook for writing codes
- Include comments and intermediate results in addition to the codes
- Export the Jupyter Notebook file in PDF format
- Turn in the PDF file at Google Classroom (late submission is not allowed)
```

#### History of git commits

```console
- Create a private repository at github 
- Commit intermediate status of working file at given steps
- Export the history of commits in PDF format
- Turn in the PDF file at Google Classroom (late submission is not allowed)
```

## Binary Classification based on Logistic Regression

> - $`(x_i, y_i)`$ denotes a pair of a training example and $`i = 1, 2, \cdots, n`$
> - $`\hat{y}_i = \sigma(z_i)`$ where $`z_i = w^T x_i + b`$ and $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$
> - The loss function is defined by $`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i(w, b)`$
> - $`f_i(w, b) = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

### 1. Plot two clusters of points for training dateset

- Generate two sets of separable random point clusters in $`\mathbb{R}^2`$
- Let $`\{ x_i \}_{i=1}^n`$ be a set of points and $`\{ y_i \}_{i=1}^n`$ be their corresponding labels
- Plot the point clusters in the training dataset using different colors depending on their labels

### 2. Plot two clusters of points for testing dataset

- Generate two sets of separable random point clusters in $`\mathbb{R}^2`$ for a testing dataset using the same centroid and the standard deviation of random generator as the training dataset
- Plot the point clusters in the testing dataset using different colors depending on their labels (different colors from the training dataset)

### 3. git commit

```console
$ git commit -a -m "Plot the training and testing datasets"
$ git push -u origin master
```

### 4. Plot the learning curves

- Apply the gradient descent algorithm
- Plot the training loss at every iteration
- Plot the testing loss at every iteration
- Plot the training accuracy at every iteration
- Plot the testing accuracy at every iteration

### 5. git commit

```console
$ git commit -a -m "Plot the learning curves"
$ git push -u origin master
```

</details>


<details>
<summary>Assignment 02</summary>
  
  # Assignment 02

```
Build a binary classifier for human versus horse based on logistic regression using the dataset that consists of human and horse images
```

## Binary classification based on logistic regression

$`(x_i, y_i)`$ denotes a pair of a training example and $`i = 1, 2, \cdots, n`$

$`\hat{y}_i = \sigma(z_i)`$ where $`z_i = w^T x_i + b`$ and $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$

The loss function is defined by $`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i(w, b)`$

$`f_i(w, b) = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using the training set
- The classifier should be tested using the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own implementation for the followings:
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- Apply the gradient descent algorithm with an appropriate learning rate
- Apply the number of iterations that lead to the convergence of the algorith
- Use the vectorization scheme in the computation of gradients and the update of the model parameters

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Plot the elapsed time at every iteration for the computation of the gradient and the update of model parameters (x-axis: iteration, y-axis: elapsed time)
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss with training and validation datasets as below:

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments [example: 20191234_02.pdf]
- A PDF file exported from the github website for the history of git commit [example: 20191234_02_git.pdf]

</details>

<details>
<summary>Assignment 03</summary>
  
  # Assignment 03

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

$`(x_i, y_i)`$ denotes a pair of a training example and $`i = 1, 2, \cdots, n`$

$`p_i = \sigma(u^T x_i + a)`$ (hidden layer)

$`q_i = \sigma(v^T p_i + b)`$ (hidden layer)

$`\hat{y}_i = \sigma(w^T q_i + c)`$ (output layer)

The logistic function $`\sigma`$ is defined by $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$

The loss function is defined by $`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i(u, v, w, a, b, c)`$

$`f_i(u, v, w, a, b, c) = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

## Neural Network Architecture

- The sizes of the hidden layers and the output layer should be determined with respect to the validation accuracy

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using the training set
- The classifier should be tested using the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own implementation for the followings:
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- Apply the gradient descent algorithm with an appropriate learning rate
- Apply the number of iterations that lead to the convergence of the algorith
- Use the vectorization scheme in the computation of gradients and the update of the model parameters

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss with training and validation datasets with your best neural network architecture as below:

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit

</details>

<details>
<summary>Assignment 04</summary>
  
  # Assignment 04

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

#### First layer

$`Z^{[1]} = W^{[1]} X + b^{[1]}`$ : $`X`$ denotes the input data

$`A^{[1]} = g^{[1]}(Z^{[1]})`$ : $`g^{[1]}`$ is the activation function at the first layer

#### Second layer

$`Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}`$

$`A^{[2]} = g^{[2]}(Z^{[2]})`$ : $`g^{[2]}`$ is the activation function at the second layer

#### Third layer

$`Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}`$

$`A^{[3]} = g^{[3]}(Z^{[3]})`$ : $`g^{[3]}`$ is the activation function at the third (output) layer

## Activation Function

- Sigmoid

    $`g(z) = \frac{1}{1 + \exp^{-z}}`$

- tanh

    $`g(z) = \frac{\exp^{z} - \exp^{-z}}{\exp^{z} + \exp^{-z}}`$

- ReLU

    $`g(z) = \max(0, z)`$

- Leaky ReLU

    $`g(z) = \max(\alpha z, z), \quad \alpha \in \mathbb{R}^+`$

## Neural Network Architecture

- The sizes of the hidden layers and the output layer should be determined with respect to the validation accuracy obtained by the network architecture with all the activation functions being sigmoid functions. ($`g^{[1]} = g^{[2]} = g^{[3]} =`$ Sigmoid)
- Apply different activation functions at all the layers except the output layer that should be Sigmoid function
- Apply different activation functions at different layers except the output layer that should be Sigmoid function

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using the training set
- The classifier should be tested using the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own functions for the followings:
    - compute the forward propagation
    - compute the backward propagation
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- Apply the gradient descent algorithm with an appropriate learning rate
- Apply the number of iterations that lead to the convergence of the algorith
- Use the vectorization scheme in the computation of gradients and the update of the model parameters

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Do not print out text message per each iteration. It should be illustrated by graphs.
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss with training and validation datasets with your best neural network architecture as below:

##### $`g^{[1]}, g^{[2]}, g^{[3]}`$ are Sigmoid (from the previous assignment)

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are tanh and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are ReLU and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are Leaky ReLU with your choice of $`\alpha`$ and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit

</details>

<details>
<summary>Assignment 05</summary>
  
  # Assignment 05

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

#### First layer

$`Z^{[1]} = W^{[1]} X + b^{[1]}`$ : $`X`$ denotes the input data

$`A^{[1]} = g^{[1]}(Z^{[1]})`$ : $`g^{[1]}`$ is the activation function at the first layer

#### Second layer

$`Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}`$

$`A^{[2]} = g^{[2]}(Z^{[2]})`$ : $`g^{[2]}`$ is the activation function at the second layer

#### Third layer

$`Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}`$

$`A^{[3]} = g^{[3]}(Z^{[3]})`$ : $`g^{[3]}`$ is the activation function at the third (output) layer

## Neural Network Architecture

- The neural network architexture should be designed to have 3 layers
- The activation function should be applied to each layer
- You can use any activation function at each layer

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using only the training set
- The classifier should be tested using only the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own functions for the followings:
    - compute the forward propagation
    - compute the backward propagation
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- You should apply the full gradient descent algorithm with your choice of learning rates
- You should apply enough number of iterations that lead to the convergence of the algorithm
- You should use the vectorization scheme in the computation of gradients and the update of the model parameters
- You can initialize the model parameters with your own algorithm

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Do not print out text message per each iteration. It should be illustrated by graphs
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss at convergence with training and validation datasets
    - training loss (at convergence)
    - validation loss (at convergence)
    - validation loss (when the best validation accuracy is achieved over all the iterations)
    - training accuracy (at convergence)
    - validation accuracy (at convergence)
    - validation accuracy (when the best validation accuracy is achieved over all the iterations)

## Grading

- The grading is given by the best validation accuracy over all the iterations (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit

</details>

<details>
<summary>Assignment 06</summary>
  
  # Assignment 06

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

## Neural Network Architecture

- The neural network architexture should be designed to have 3 layers
- The activation function should be applied to each layer
- Sigmoid function is used for an activation function at each layer

#### First layer

$`Z^{[1]} = W^{[1]} X + b^{[1]}`$ : $`X`$ denotes the input data

$`A^{[1]} = g(Z^{[1]})`$

#### Second layer

$`Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}`$

$`A^{[2]} = g(Z^{[2]})`$

#### Third layer

$`Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}`$

$`A^{[3]} = g(Z^{[3]})`$

## Activation Function

- Sigmoid : 
    $`g(z) = \frac{1}{1 + \exp^{-z}}`$

## Loss function with a regularization term based on $`L_2^2`$ norm

$`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i + \frac{\lambda}{2} \left( \| W^{[1]} \|_F^2 + \| W^{[2]} \|_F^2 + \| W^{[3]} \|_F^2 \right)`$

- Cross Entropy : 
    $`f_i = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

- Frobenius Norm : 
    $`\| W \|_F = \left( \sum_i \sum_j w_{ij}^2 \right)^{\frac{1}{2}}`$

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using only the training set
- The classifier should be tested using only the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own functions for the followings:
    - compute the forward propagation
    - compute the backward propagation
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- You should apply the full gradient descent algorithm with your choice of learning rates
- You should apply enough number of iterations that lead to the convergence of the algorithm
- You should use the vectorization scheme in the computation of gradients and the update of the model parameters
- You can initialize the model parameters with your own algorithm

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Do not print out text message per each iteration. It should be illustrated by graphs
- Demonstrate the role of regularization with varying parameter $`\lambda`$ for the tradeoff between bias and variance
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss at convergence with training and validation datasets
    - training loss (at convergence)
    - validation loss (at convergence)
    - training accuracy (at convergence)
    - validation accuracy (at convergence)

##### Bias (large $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### Variance (small $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### Best Generalization (appropriate $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Grading

- The grading is given by the validation accuracy for the best generalization (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit
</details>

<details>
  
<summary>Assignment 07</summary>

# Assignment 07

```
Build a binary classifier based on fully connected layers for the human versus horse dataset using pytorch library 
```

## Binary classification based on fully connected neural network

## Neural Network Architecture

- Build a neural network model based on the fully connected layers with pytorch library
- You can determine the number of layers
- You can determine the size of each layer
- You can determine the activation function at each layer except the output layer
- You use the sigmoid function for the activation fuction at the output layer

## Loss function with a regularization term based on $`L_2^2`$ norm

$`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i + \frac{\lambda}{2} \left( \| W \|_2^2 \right)`$

- Cross Entropy : 
    $`f_i = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$, where $`y_i`$ denotes the ground truth and $`\hat{y}_i`$ denotes the output of the network

- Regularization : 
    $`\| W \|_2^2 = \left( \sum_i w_{i}^2 \right)`$, where $`w_{i}`$ denotes all the model parameters

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using only the training set
- The classifier should be tested using only the validation set

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You should use pytorch library for the construction of the model and the optimization

### Neural Network Model in pytorch (Linear.py)

```python
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):

    def __init__(self, num_classes=2):

        super(Linear, self).__init__()

        self.number_class   = num_classes

        _size_image     = 100* 100
        _num1           = 50
        _num2           = 50
        
        self.fc1        = nn.Linear(_size_image, _num1, bias=True)
        self.fc2        = nn.Linear(_num1, _num2, bias=True)
        self.fc3        = nn.Linear(_num2, num_classes, bias=True)

        self.fc_layer1  = nn.Sequential(self.fc1, nn.ReLU(True))
        self.fc_layer2  = nn.Sequential(self.fc2, nn.ReLU(True))
        self.fc_layer3  = nn.Sequential(self.fc3, nn.ReLU(True))
        
        self.classifier = nn.Sequential(self.fc_layer1, self.fc_layer2, self.fc_layer3)
        
        self._initialize_weight()        
        
    def _initialize_weight(self):

        for m in self.modules():
            
            n = m.in_features
            m.weight.data.uniform_(- 1.0 / math.sqrt(n), 1.0 / math.sqrt(n))

            if m.bias is not None:

                m.bias.data.zero_()

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
```

### Training and Testing in pytorch (main.py)

```python
# -----------------------------------------------------------------------------
# import packages
# -----------------------------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import sys
import os
import numpy as np
import time
import datetime 
import csv
import configparser
import argparse
import platform

from torchvision import datasets, transforms
from torch.autograd import Variable
from random import shuffle

# -----------------------------------------------------------------------------
# load dataset
# -----------------------------------------------------------------------------

set_train   = 
set_test    = 

num_classes = 2

# -----------------------------------------------------------------------------
# load neural network model
# -----------------------------------------------------------------------------

from Linear import *
    model = Linear(num_classes=num_classes)

# -----------------------------------------------------------------------------
# Set the flag for using cuda
# -----------------------------------------------------------------------------

bCuda = 1

if bCuda:
 
    model.cuda()

# -----------------------------------------------------------------------------
# optimization algorithm
# -----------------------------------------------------------------------------

optimizer   = optim.SGD(model.parameters())
objective   = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# function for training the model
# -----------------------------------------------------------------------------

def train():

    # print('train the model at given epoch')

    loss_train          = []

    model.train()

    for idx_batch, (data, target) in enumerate(loader_train):

        if bCuda:
        
            data, target    = data.cuda(), target.cuda()

        data, target    = Variable(data), Variable(target)

        optimizer.zero_grad()

        output  = model(data)
        loss    = objective(output, target)

        loss.backward()
        optimizer.step()

        loss_train_batch    = loss.item() / len(data)
        loss_train.append(loss_train_batch)
        
    loss_train_mean     = np.mean(loss_train)
    loss_train_std      = np.std(loss_train)

    return {'loss_train_mean': loss_train_mean, 'loss_train_std': loss_train_std}

# -----------------------------------------------------------------------------
# function for testing the model
# -----------------------------------------------------------------------------

def test():

    # print('test the model at given epoch')

    accuracy_test   = []
    loss_test       = 0
    correct         = 0

    model.eval()

    for idx_batch, (data, target) in enumerate(loader_test):

        if bCuda:
        
            data, target    = data.cuda(), target.cuda()

        data, target    = Variable(data), Variable(target)

        output  = model(data)
        loss    = objective(output, target)

        loss_test   += loss.item()
        pred        = output.data.max(1)[1]
        correct     += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_test       = loss_test / len(loader_test.dataset)
    accuracy_test   = 100. * float(correct) / len(loader_test.dataset)

    return {'loss_test': loss_test, 'accuracy_test': accuracy_test}

# -----------------------------------------------------------------------------
# iteration for the epoch
# -----------------------------------------------------------------------------

for e in range(epoch):
        
    result_train    = train()
    result_test     = test()

    loss_train_mean[e]  = result_train['loss_train_mean']
    loss_train_std[e]   = result_train['loss_train_std']
    loss_test[e]        = result_test['loss_test']
    accuracy_test[e]    = result_test['accuracy_test']
```


## Optimization

- You can use weight decay option in the pytorch optimization function
- You can use mini-batch gradient descent (stochastic gradient descent) with your choice of mini-batch size
- You can use a different learning rate at each iteration
- You can initialize the values of the model parameters with your choice of algorithm
- You should apply enough number of iterations that lead to the convergence of the algorithm

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

#### Output (text)

- Print out the followings at each epoch
    - average training loss within the mini-batch cross iterations in the training data
    - average training accuracy within the mini-batch cross iterations in the training data
    - testing loss using the testing data at each epoch
    - testing accracy using the testing data at each epoch
    - [epoch #####] loss: (training) #####, (testing) #####, accuracy: (training) #####, (testing) #####

#### Output (graph)

- Plot the average of the training loss within the mini-batch cross iterations
- Plot the standard deviation of the training loss withint the mini-batch cross iterations
- Plot the average of the training accuracy within the mini-batch cross iterations
- Plot the standard deviation of the training accuracy withint the mini-batch cross iterations
- Plot the testing loss at each epoch
- Plot the testing accuracy at each epoch

#### Output (table)

- Present the final loss and accuracy at convergence

| dataset    | loss       | accuracy   |
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Grading

- The grading is given by the validation accuracy for the best generalization (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit
</details>

<details>
<summary>Assignment 08</summary>
  
 # Assignment 08

```
Develop a denoising algorithm based on an auto-encoder architecture using pytorch library in the supervised learning framework 
```

## Image denoising problem

- Denoising aims to reconstruct a clean image from a noisy observation
- We use a simple additive noise model using the Normal distribution:

    $`f = u + \eta`$
    
    where $`f`$ denotes a noisy observation, $`u`$ denotes a desired clean reconstruction, and $`\eta`$ denotes a noise process following the normal distribution:

    $`\eta \sim N(0, \sigma^2)`$

    where $`N(0, \sigma^2)`$ denotes the normal distribution with mean 0 and standard deviation $`\sigma`$

## Neural Network Architecture

- Build an auto-encoder architecture based on the convolutional neural network using pytorch
- The dimension of the network input should be the same as the dimension of the network output
- You can design your neural network architecture as you want

## Loss function 

- You can design your loss function for computing a dissimilarity between the output and the ground truth
- The evaluation of the algorithm is given by the mean squared error:

    $`\ell(h, \hat{h}) = \| h - \hat{h} \|_2^2`$

    where $`h`$ denotes a clean ground truth and $`\hat{h}`$ denotes an output of the network

## Dataset

- The dataset consists of training and testing images that are small pieces taken from images 
- The dimension of image is 120x80
- The number of training images is 4400
- The number of testing images is 400
- The range of training images is [0.2, 0.8]
- The range of testing images is [0.0601, 0.9744]
- The training images are clean
- Test testing images are noisy
- The ground truth for the noisy testing images is not given
- The ground truth for the noisy testing images is used for the evalution
- The noise levels of the testing images are 0.01, 0.02, 0.03 and 0.04
- Example images are shown with different degrees of noise $`\sigma = 0.01, 0.02, 0.03, 0.04`$ from the left as below:

![](img/std_0.01_clean1.png) ![](img/std_0.02_clean1.png) ![](img/std_0.03_clean1.png)  
![](img/std_0.01_noise1.png) ![](img/std_0.02_noise1.png) ![](img/std_0.03_noise1.png)  
  
![](img/std_0.04_clean1.png) ![](img/std_0.01_clean2.png) ![](img/std_0.02_clean2.png)  
![](img/std_0.04_noise1.png) ![](img/std_0.01_noise2.png) ![](img/std_0.02_noise2.png)  
  
![](img/std_0.03_clean2.png) ![](img/std_0.04_clean2.png)  
![](img/std_0.03_noise2.png) ![](img/std_0.04_noise2.png)  

## Implementation

- Write codes in python programming
- Use pytorch libarary
- Use ```jupyter notebook``` for the programming environment
- You can use any python libraries
- Write your own code for your neural network architecture
- Write your own code for the training procedure
- Write your own code for the testing procedure

## Code for reading and writing image data

```python
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# custom dataloader for .npy file
class numpyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    NUM_EPOCH       = 2
    
    transform       = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                ])

    # for training
    traindata       = np.load('train.npy')
    traindataset    = numpyDataset(traindata, transform)
    trainloader     = DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=2)

    for epoch in range(NUM_EPOCH):
        for batch_idx, data in enumerate(trainloader):
            image   = data[0]
            to_img  = transforms.ToPILImage()
            image   = to_img(image)

            fig     = plt.figure()
            ax      = fig.add_subplot(1, 1, 1)
            ax.imshow(image, cmap='gray')

            '''
            your code for train
            '''

    # for testing
    testdata        = np.load('test.npy')
    testdataset     = numpyDataset(testdata, transform)
    testloader      = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=2)

    result_for_submit = None    # this is for submit file

    for batch_idx, data in enumerate(testloader):

        result_of_test = data

        if batch_idx == 0:
            result_for_submit = result_of_test
        else:
            try:
                result_for_submit = torch.cat([result_for_submit, result_of_test], dim=0)

            except RuntimeError:
                transposed = torch.transpose(result_of_test, 2, 3)
                result_for_submit = torch.cat([result_for_submit, transposed], dim=0)
        
    # the submit_file.shape must be (400,1,120,80) 
    submit_file = result_for_submit.detach().numpy()
    np.save('your_name.npy', submit_file)
```

## Optimization

- You can use any optimization techniques

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

#### Output (text)

- Print out the followings at each epoch
    - The average of the training loss over mini-batch iterations at each epoch
    - [epoch #####] loss: (training) ########

#### Output (graph)

- Plot the average of the training loss over mini-batch iterations at each epoch
- Plot the standard deviation of the training loss over mini-batch iterations at each epoch

#### Output (file)

- Save the output of the network for the given training images as a file

## Grading

- The grading is given by the performance of the algorithm based on the evaluation criterion (mean squared error) among the complete ones
    - up to top 25% : score 10
    - up to top 50% : score 8
    - up to top 75% : score 6
    - up to top 100% : score 4
    - incomplete : maximum 3
    
## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit
- A data file of the denoising results for the testing images (give a filename: yourname.npy) 
</details>

