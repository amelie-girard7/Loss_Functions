# Understanding Statistical Foundations: MLE, NLL, Cross-Entropy, and K-L Divergence

This notebook aims to provide a comprehensive understanding of the key statistical concepts like Maximum Likelihood Estimation (MLE), Negative Log-Likelihood (NLL), Cross Entropy, and Kullback-Leibler (K-L) Divergence.

## Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
    - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
    - [Negative Log-Likelihood (NLL)](#negative-log-likelihood-nll)
    - [Cross-Entropy](#cross-entropy)
    - [Kullback-Leibler (K-L) Divergence](#kullback-leibler-k-l-divergence)
2. [Equivalence of NLL, Cross-Entropy, and K-L Divergence](#equivalence-of-nll-cross-entropy-and-k-l-divergence)
3. [Computational Advantages and Optimization Techniques](#computational-advantages-and-optimization-techniques)
4. [Academic Papers](#academic-papers)

## 1. Theoretical Foundations <a name="theoretical-foundations"></a>

### Maximum Likelihood Estimation (MLE) <a name="maximum-likelihood-estimation-mle"></a>

Maximum Likelihood Estimation is a method used for estimating the parameters of a statistical model. Given a set of observations $( \mathbf{X} = \{x_1, x_2, \ldots, x_n\} )$ and a parametric model $( f(x; \theta) )$ that describes how the data is generated, MLE aims to find the parameter $( \theta )$ that maximizes the likelihood function $( L(\theta; \mathbf{X}) )$.


#### Mathematical Formulation

The likelihood function $( L(\theta; \mathbf{X}) )$ is defined as the joint probability of observing the given data $( \mathbf{X} )$ under the model parameterized by $( \theta $):

$[
L(\theta; \mathbf{X}) = P(\mathbf{X} | \theta) = \prod_{i=1}^{n} f(x_i; \theta)
]$

The MLE aims to find $( \theta $) that maximizes this likelihood function:

$[
\hat{\theta}_{MLE} = \arg \max_{\theta} L(\theta; \mathbf{X})
]$

Often, it is more convenient to work with the log-likelihood function, $( \ell(\theta; \mathbf{X}) $), which is the natural logarithm of the likelihood function:

$[
\ell(\theta; \mathbf{X}) = \log L(\theta; \mathbf{X}) = \sum_{i=1}^{n} \log f(x_i; \theta)
]$

The likelihood function $( L(\theta; \mathbf{X}) $) is a way to measure how well a particular set of parameters $( \theta $) explains the observed data $( \mathbf{X} $). In simpler terms, it tells us how "likely" the observed data is, given a specific model and its parameters.

The function $( f(x_i; \theta) $) represents the probability of observing a single data point $( x_i $) given the parameters $( \theta $). The likelihood function is the product of these probabilities for all observed data points. In essence, it's a way to combine the probabilities for each individual data point into a single measure that tells us how well the model explains all the data points together.

Maximum Likelihood Estimation (MLE) aims to find the set of parameters $( \theta $) that makes the observed data most "likely" under the model. In other words, MLE finds the $( \theta $) that maximizes the likelihood function $( L(\theta; \mathbf{X}) $).

So, MLE is like tuning the knobs of a machine (the model) until it performs at its best (maximizes the likelihood) for a given task (explaining the observed data).

#### Objective
The objective of MLE is to find the parameter $( \theta $) that makes the observed data most probable. In other words, MLE finds the parameter values that maximize the likelihood of making the observed data.

#### Optimization

The optimization is usually performed using iterative methods like Newton-Raphson or gradient-based methods like stochastic gradient descent, especially when dealing with high-dimensional data or complex models.

#### Assumptions and Limitations

- MLE assumes that the data is independently and identically distributed (i.i.d.).
- It may be sensitive to outliers.
- It may overfit the data if the model is too complex.

#### Practical examples

##### MLE with Gaussian Distribution
In this example, we'll generate some synthetic data from a Gaussian distribution and then use MLE to estimate its parameters ($( \mu $) and $( \sigma $)).
[Link to the notebook](..\src\Maximum_Likelihood.ipynb)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.stats import norm

# Generate synthetic data
np.random.seed(0)
true_mean = 5
true_std = 2
n_samples = 1000

# Generate synthetic data
data = np.random.normal(true_mean, true_std, n_samples)

# Visualize Synthetic Data
# Creating a histogram to offer an initial look at the data's distribution, aiding in understanding before applying MLE.
plt.hist(data, bins=30, density=True, alpha=0.5, color='g', label='Generated Data')
plt.title('Frequency Distribution of Generated Data')
plt.xlabel('Data Points')
plt.ylabel('Relative Frequency')
plt.legend()
plt.show()

# Define the negative log-likelihood function
def neg_log_likelihood(params):
    mean, std = params
    nll = -np.sum(np.log(1 / (np.sqrt(2 * np.pi) * std)) + 
                  (-((data - mean) ** 2) / (2 * std ** 2)))
    return nll

# Perform MLE using optimization
initial_guess = [1, 1]
result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', bounds=[(None, None), (0.01, None)])

# Extract the MLE estimates
mle_mean, mle_std = result.x

# Display the MLE mean and standard deviation
print(f"True mean: {true_mean}, MLE estimated mean: {mle_mean}")
print(f"True standard deviation: {true_std}, MLE estimated standard deviation: {mle_std}")

# Plot the data and the estimated Gaussian distribution
plt.hist(data, bins=30, density=True, alpha=0.5, label='Observed Data')

x = np.linspace(min(data), max(data), 100)
pdf = (1 / (mle_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mle_mean) / mle_std)**2)
plt.plot(x, pdf, label='MLE Estimated Distribution')

plt.title('MLE for Gaussian Distribution')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.show()
```

### Negative Log-Likelihood (NLL)

Negative Log-Likelihood is a loss function commonly used in statistical modeling and machine learning, particularly for classification problems. The NLL quantifies how well a set of predicted probabilities explains the observed outcomes, with lower values indicating better-fitted models.

#### Mathematical Formulation

The NLL is derived from the likelihood function $( L(\theta; \mathbf{X}) $), and is given by:

$[
\text{NLL}(\theta; \mathbf{X}) = - \log L(\theta; \mathbf{X})
$]

For a set of $( N $) observations, the NLL can be represented as:

$[
\text{NLL}(\theta; \mathbf{X}) = -\sum_{i=1}^{N} \log f(x_i; \theta)
$]

#### Objectives of minimising NLL

The aim is to find the model parameters $( \theta $) that minimize the NLL. In essence, minimizing NLL is equivalent to maximizing the likelihood of the model, making the observed data most probable.

#### Application in Classification Problems

In classification tasks, NLL is often used with softmax activation in the output layer. The NLL for a multi-class classification problem can be given as:

$[
\text{NLL} = -\sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log(p_{ij})
$]

Here, $( N $) is the number of samples, $( K $) is the number of classes, $( y_{ij} $) is the true label, and $( p_{ij} $) is the predicted probability.

#### Advantages and Limitations
Advantages: Provides a probabilistic interpretation, well-suited for classification problems.
Limitations: Can be sensitive to outliers, assumes that the model and data are well-described by the likelihood function.


#### Practical examples
[Link to the notebook](..\src\Negative_Log_Likelihood.ipynb)

This example demonstrates the concept of Negative Log-Likelihood (NLL) in the context of a simple Natural Language Generation (NLG) task. In this example, we'll use a toy dataset and a basic model to predict the next word in a sequence.

- Data Preparation: We have a toy dataset where each sequence is a list of words. The last word in each sequence is what we want to predict.

- Model: We use a simple RNN model for this task. The model takes the word indices as input, embeds them into a continuous space, passes them through an RNN layer, and finally through a fully connected layer to predict the next word.

- Loss Function: We use the CrossEntropyLoss, which is a generalization of NLL loss suitable for multi-class classification.

- Training: We train the model using the Adam optimizer. The model parameters are updated to minimize the NLL.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Toy dataset: Each row is a sequence, and the last element is the target word
data = [
    ['hello', 'how', 'are', 'you'],
    ['I', 'am', 'fine'],
    ['hello', 'I', 'am', 'Amelie']
]
word_to_idx = {'<PAD>': 0, 'hello': 1, 'how': 2, 'are': 3, 'you': 4, 'I': 5, 'am': 6, 'fine': 7, 'Amelie': 8}
vocab_size = len(word_to_idx)

# Convert words to integers
data_idx = [[word_to_idx[w] for w in seq] for seq in data]

# Prepare data for training
X_train = [torch.tensor(seq[:-1], dtype=torch.long) for seq in data_idx]
y_train = [torch.tensor(seq[-1], dtype=torch.long) for seq in data_idx]

 model
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

# Hyperparameters
embed_size = 10
hidden_size = 20
learning_rate = 0.001

# Initialize model, loss, and optimizer
model = SimpleRNN(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()  # NLL loss is a specific case of CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(100):
    for x, y in zip(X_train, y_train):
        x = x.view(-1, 1)
        output = model(x)
        
        # Repeat y to match the batch size
        y_repeated = y.repeat(output.shape[0])
        
        loss = criterion(output, y_repeated)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print("Training complete.")
```


### Cross-Entropy
Cross-Entropy is a loss function commonly used in machine learning and optimization, particularly for classification tasks. It measures the dissimilarity between the true distribution $( P $) and the predicted distribution $( Q $) over the same events. Lower values of Cross-Entropy indicate better performance.

#### Mathematical Formulation

For two discrete probability distributions $( P $) and $( Q $), the Cross-Entropy $( H(P, Q) $) is defined as:

$[
H(P, Q) = -\sum_{x} P(x) \log(Q(x))
$]

For a classification problem with $( N $) samples and $( C $) classes, the Cross-Entropy loss $( L $) is given by:

$[
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})
$]

Here, $( y_{ic} $) is the true label and $( p_{ic} $) is the predicted probability for class $( c $) of sample $( i $).

#### Objective of Minimizing Cross-Entropy

The goal is to find the model parameters that minimize the Cross-Entropy loss. This effectively makes the predicted distribution $( Q $) as close as possible to the true distribution $( P $).


#### Advantages and Limitations

Advantages: Provides a measure of similarity between the true and predicted distributions, well-suited for classification problems.
Limitations: Like other loss functions, Cross-Entropy can also be sensitive to outliers and may suffer from numerical instability for incorrect predictions with high confidence.

#### Example

Here's a Python code snippet to compute the Cross-Entropy loss using NumPy:

```python
import numpy as np

# True labels in one-hot encoding
y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

# Predicted probabilities
y_pred = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

# Compute Cross-Entropy
cross_entropy = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

print(f"Cross-Entropy: {cross_entropy}")
```

Output:
Cross-Entropy: 0.1738073366910675

### Kullback-Leibler (K-L) Divergence

Kullback-Leibler (K-L) Divergence is a measure used in information theory to quantify the difference between two probability distributions $( P $) and $( Q $). Unlike Cross-Entropy, it specifically measures how one distribution diverges from a reference distribution.

#### Mathematical Formulation

For two discrete probability distributions $( P $) and $( Q $), the K-L Divergence $( D_{KL}(P \parallel Q) $) is defined as:

$[
D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$]

#### Objective of Minimizing K-L Divergence

The goal of minimizing K-L Divergence is to make the approximating distribution $( Q $) as close as possible to the target distribution $( P $).

#### Application in Machine Learning

K-L Divergence is often used in unsupervised learning, particularly in methods like t-SNE for dimensionality reduction and in variational inference.

#### Advantages and Limitations

- **Advantages**: Provides a way to measure the divergence of one distribution from another, useful in various machine learning applications.
- **Limitations**: Like other loss functions and measures, K-L Divergence can be sensitive to outliers. It is also not symmetric, meaning $( D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P) $).


#### Practical examples

Here's a Python code snippet to compute the K-L Divergence using NumPy:

```python
import numpy as np

# True distribution P
p = np.array([0.2, 0.4, 0.4])

# Approximating distribution Q
q = np.array([0.1, 0.7, 0.2])

# Compute K-L Divergence
kl_divergence = np.sum(p * np.log(p / q))

print(f"K-L Divergence: {kl_divergence}")

```

Output:
K-L Divergence: 0.09151622184943575



### 2. Equivalence of NLL, Cross-Entropy, and K-L Divergence in Classification Problems

 **Unveiling the Identity of NLL and Cross-Entropy**: Negative Log-Likelihood (NLL) and Cross-Entropy serve as cornerstone loss functions in machine learning, particularly in classification tasks and Natural Language Generation (NLG) models. These two loss functions are not just similar; they are mathematically identical when it comes to classification problems. This identity often goes unnoticed due to the different contexts in which each is typically introduced—Cross-Entropy often appears in binary classification scenarios, while NLL is more universally applied. Both aim to optimize the same objective function and reach their minimum or maximum values at identical points. For an in-depth exploration and practical demonstration, consult the Jupyter Notebook: [Comparison of NLL and Cross-Entropy](../src/Comparaision_NLL_Enthropy.ipynb).

 **Connection to Kullback-Leibler Divergence**: Intriguingly, when the ground truth labels are binary (either 0 or 1), the NLL loss function becomes a specific instance of Kullback-Leibler (K-L) Divergence. This is a widely-used metric to quantify the divergence between two probability distributions. In the context of binary classification, NLL, Cross-Entropy, and K-L Divergence essentially measure the same "distance" between the predicted and actual data distributions. For further insights, you may refer to this enlightening [article](http://www.awebb.info/probability/2017/05/18/cross-entropy-and-log-likelihood.html).

### 3. Computational Advantages and Optimization Techniques

#### Computational Efficiency of NLL
Negative Log-Likelihood (NLL) is often favored over direct likelihood calculations for computational reasons. Calculating the likelihood involves taking the product of probabilities, which can result in numerical underflow issues, especially when dealing with a large number of samples. NLL, on the other hand, works with the sum of log-probabilities, making it computationally more stable and efficient. This is particularly beneficial in high-dimensional spaces and large datasets, common scenarios in machine learning applications.

#### Optimization Using Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is a widely-used optimization algorithm for minimizing NLL. The algorithm updates the model parameters iteratively, using a subset of the data, making it more computationally efficient than batch gradient descent. This is especially useful for large-scale problems where using the entire dataset for each update is computationally expensive. The stochastic nature of the algorithm also helps escape local minima, providing a more robust optimization process.


## Academic Papers
I found some academic papers that could help deepen our understanding of the relationship between Negative Log-Likelihood, Maximum Likelihood Estimation, Cross Entropy, and Kullback-Leibler Divergence:

**The Estimation of Distributions and the Minimum Relative Entropy Principle**
- **Authors**: H. Mühlenbein, Robin Höns
- **Abstract**: This paper discusses the relationship of Estimation of Distribution Algorithms (EDA) to algorithms in statistics and artificial intelligence. It focuses on the role of maximum entropy approximations and Kullback-Leibler divergence.
- **Cited by**: 97
- [PDF](http://www.muehlenbein.org/minrel.PDF)
- [Landing Page](https://dx.doi.org/10.1162/1063656053583469).

**Adaptation in log-concave density estimation**
- **Authors**: Arlene K. H. Kim, Adityanand Guntuboyina, R. Samworth
- **Abstract**: The paper discusses the log-concave maximum likelihood estimator of a density and its rate of convergence with respect to various global loss functions, including Kullback–Leibler divergence.
- **Cited by**: 45
- [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-46/issue-5/Adaptation-in-log-concave-density-estimation/10.1214/17-AOS1619.pdf)
- [Landing Page](https://dx.doi.org/10.17863/CAM.11980).


**Entropy and Divergence Associated with Power Function and the Statistical Application**
- **Authors**: S. Eguchi, Shogo Kato
- **Abstract**: This paper discusses the relationship between entropy and the maximum likelihood method, focusing on Kullback-Leibler divergence.
- **Cited by**: 44
- [PDF](https://www.mdpi.com/1099-4300/12/2/262/pdf?version=1424784678)
- [Landing Page](https://dx.doi.org/10.3390/e12020262).

**Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution**
- **Authors**: Akira Ito, Kotaro Saito, Rei Ueno, N. Homma
- **Abstract**: This paper discusses the problems caused by data imbalance in deep learning-based attacks and introduces Kullback–Leibler divergence as a metric to measure this effect.
- **Cited by**: 18
- [PDF](https://ieeexplore.ieee.org/ielx7/10206/9151439/09464254.pdf)
- [Landing Page](https://dx.doi.org/10.1109/TIFS.2021).


**Entropies and rates of convergence for maximum likelihood and Bayes estimation for mixtures of normal densities**  
   - **Authors**: S. Ghosal, A. Vaart  
   - **Publication Date**: 2001-10-01  
   - **Cited By**: 320  
   - **Abstract**: This paper studies the rates of convergence of the MLE and posterior distribution in density estimation problems. It also discusses Kullback-Leibler type neighborhoods.  
   - [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Entropies-and-rates-of-convergence-for-maximum-likelihood-and-Bayes/10.1214/aos/1013203452.pdf)
   - [Landing Page](https://dx.doi.org/10.1214/AOS/1013203452).

**APPROXIMATION OF DENSITY FUNCTIONS BY SEQUENCES OF EXPONENTIAL FAMILIES**  
   - **Authors**: A. Barron, Chyong-Hwa Sheu  
   - **Publication Date**: 1991-09-01  
   - **Cited By**: 259  
   - **Abstract**: This paper discusses the estimation of a probability density function using maximum likelihood and relates it to the principle of maximum entropy or minimum relative entropy.  
   - [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-3/Approximation-of-Density-Functions-by-Sequences-of-Exponential-Families/10.1214/aos/1176348252.pdf)
   - [Landing Page](https://dx.doi.org/10.1214/AOS/1176348252).

**Spectral entropies as information-theoretic tools for complex network comparison**  
   - **Authors**: M. Domenico, J. Biamonte  
   - **Publication Date**: 2016-09-05  
   - **Cited By**: 127  
   - **Abstract**: This paper uses techniques inspired by quantum statistical mechanics to define an entropy measure for complex networks and discusses Kullback-Leibler divergence.  
   - [PDF](http://link.aps.org/pdf/10.1103/PhysRevX.6.041062)
   - [Landing Page](https://dx.doi.org/10.1103/PhysRevX.6.041062).

**Model selection principles in misspecified models**  
   - **Authors**: Jinchi Lv, Jun S. Liu  
   - **Publication Date**: 2010-05-29  
   - **Cited By**: 120  
   - **Abstract**: This paper discusses model selection principles like Bayesian principle and Kullback-Leibler divergence in misspecified models.  
   - [PDF](https://arxiv.org/pdf/1005.5483)
   - [Landing Page](https://dx.doi.org/10.1111/rssb.12023).

