# Understanding Statistical Foundations: MLE, NLL, Cross-Entropy, and K-L Divergence

This notebook aims to provide a comprehensive understanding of the difference between the  Negative Log-Likelihood (NLL), Cross Entropy, and Kullback-Leibler (K-L) Divergence.

## Table of Contents
1. [Theoretical Foundations](#1-theoretical-foundations)
    - [1.1 Maximum Likelihood Estimation (MLE)](#11-maximum-likelihood-estimation-mle)
    - [1.2 Negative Log-Likelihood (NLL)](#12-negative-log-likelihood-nll)
    - [1.3 Cross-Entropy](#13-cross-entropy)
    - [1.4 Kullback-Leibler (K-L) Divergence](#14-kullback-leibler-k-l-divergence)
    - [1.5 Optimal Transport (OT)](#15-Optimal-Transport)
2. [Equivalence of NLL, Cross-Entropy, and K-L Divergence](#2-equivalence-of-nll-cross-entropy-and-k-l-divergence)
3. [Computational Advantages and Optimization Techniques](#3-computational-advantages-and-optimization-techniques)
4. [Conclusion](#4-Conclusion)
5. [Academic References](#5-academic-references)

## 1. Theoretical Foundations <a name="theoretical-foundations"></a>

### 1.1 Maximum Likelihood Estimation (MLE) <a name="maximum-likelihood-estimation-mle"></a>

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

#### Objectives
The objective of MLE is to find the parameter $( \theta $) that makes the observed data most probable. In other words, MLE finds the parameter values that maximize the likelihood of making the observed data.

The optimization is usually performed using iterative methods like Newton-Raphson or gradient-based methods like stochastic gradient descent, especially when dealing with high-dimensional data or complex models.

#### Assumptions and Limitations

- MLE assumes that the data is independently and identically distributed (i.i.d.).
- It may be sensitive to outliers.
- It may overfit the data if the model is too complex.

#### Practical examples: MLE with Gaussian Distribution
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

### 1.2 Negative Log-Likelihood (NLL)

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

#### Objectives 

The aim is to find the model parameters $( \theta $) that minimize the NLL. In essence, minimizing NLL is equivalent to maximizing the likelihood of the model, making the observed data most probable.


In classification tasks, NLL is often used with softmax activation in the output layer. The NLL for a multi-class classification problem can be given as:

$[
\text{NLL} = -\sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log(p_{ij})
$]

Here, $( N $) is the number of samples, $( K $) is the number of classes, $( y_{ij} $) is the true label, and $( p_{ij} $) is the predicted probability.

#### Advantages and Limitations
Advantages: Provides a probabilistic interpretation, well-suited for classification problems.
Limitations: Can be sensitive to outliers, assumes that the model and data are well-described by the likelihood function.


#### Practical examples
[Execute](..\src\Negative_Log_Likelihood.ipynb)

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


### 1.3 Cross-Entropy
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

#### Objective

The goal is to find the model parameters that minimize the Cross-Entropy loss. This effectively makes the predicted distribution \( Q \) as close as possible to the true distribution $( P $).


#### Advantages and Limitations

Advantages: Provides a measure of similarity between the true and predicted distributions, well-suited for classification problems.
Limitations: Like other loss functions, Cross-Entropy can also be sensitive to outliers and may suffer from numerical instability for incorrect predictions with high confidence.

#### Practical Example

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

### 1.4 Kullback-Leibler (K-L) Divergence

Kullback-Leibler (K-L) Divergence is a measure used in information theory to quantify the difference between two probability distributions $( P $) and $( Q $). Unlike Cross-Entropy, it specifically measures how one distribution diverges from a reference distribution.

#### Mathematical Formulation

For two discrete probability distributions $( P $) and $( Q $), the K-L Divergence $( D_{KL}(P \parallel Q) $) is defined as:

$[
D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$]

#### Objective

The goal of minimizing K-L Divergence is to make the approximating distribution $( Q $) as close as possible to the target distribution $( P $).

K-L Divergence is often used in unsupervised learning, particularly in methods like t-SNE for dimensionality reduction, and in variational inference.

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


### 1.4 Optimal Transport (OT)

Optimal Transport (OT) is a mathematical framework that seeks to find the most efficient way to transport goods between multiple suppliers and consumers, given a certain cost metric. In the context of machine learning and particularly Natural Language Generation (NLG), OT serves as an alternative loss function to Negative Log-Likelihood (NLL) or Maximum Likelihood Estimation (MLE).

#### Mathematical Background

Given two weighted sets $( A = \{(a_1, w_1),...(a_N,w_N)\} $) and $( B = \{(b_1, w_1),...(b_M,w_M)\} $), OT aims to find a matching that minimizes the overall cost. The cost is often represented by a distance metric $( d(a_i, b_j) $) between the elements $( a_i $) and $( b_j $). The OT problem can be formulated as:

$[
\text{Minimize} \sum_{i=1}^{N} \sum_{j=1}^{M} d(a_i, b_j) \cdot T_{ij}
$]

subject to certain constraints that ensure the mass preservation of both sets $( A $) and $( B $).

#### Objective

The primary objective of employing Optimal Transport (OT) in the context of Natural Language Generation (NLG) is to serve as a sequence-level loss function that offers a geometrically meaningful way to compare probability distributions of word embeddings. OT provides a robust and efficient mechanism to measure the "distance" between two distributions, which in the case of NLG, would typically be the ground truth and the model's output.

The Optimal Transport problem is mathematically formulated as minimizing the following objective function:

$[
\text{OT Loss} = W_2^2(\text{Predicted Distribution}, \text{Ground Truth Distribution})
$]

Here, $( W_2^2 $) represents the 2-Wasserstein distance, a specific instance of the Wasserstein distance metric. The optimal transport plan $( T $) is computed to minimize this loss, and the model is trained to minimize the OT loss during the training process.

In the practical context, given two sentences (or sequences), their word-level or phrase-level embeddings are used to form the distributions $( \mu $) and $( \nu $). A cost matrix $( C $) is computed, which contains the pairwise distances between the elements of $( \mu $) and $( \nu $), based on a chosen distance metric (e.g., squared Euclidean distance). The OT plan $( T $) is then computed, and the OT loss is used as a sequence-level loss for training the model.

#### Advantages and Limitations

- **Advantages**: OT provides a geometrically meaningful way to compare probability distributions, which can be particularly useful when the support of the distributions do not align perfectly.
- **Limitations**: Computational complexity can be a concern, particularly for large datasets and high dimensions.

#### Practical example
Let's dissect Optimal Transport (OT) with a concrete example inspired by the paper ( Add Link to the paper). In the paper, OT is used as a sequence-level loss for sequence-to-sequence models. The idea is to match each word in the target sentence with a word in the generated sentence in such a way that the total "transportation cost" is minimized.

##### step 1. Embedding Space
- Assume we have word embeddings for each word in both the target and generated sentences. These embeddings exist in some high-dimensional space (e.g., 300 dimensions for GloVe or Word2Vec).

##### step 2. Cost Matrix
- Compute a cost matrix where each entry $([i, j]$) represents the "distance" between the $(i$)-th word in the target sentence and the $(j$)-th word in the generated sentence.

##### step 3. Optimal Assignment
- Use an algorithm (like the Hungarian algorithm) to find the optimal assignment that minimizes the total transportation cost. This assignment pairs each word in the target sentence with a word in the generated sentence in a way that the total "distance" (or cost) is minimized.
[Execute](..\src\Optimal_transport.ipynb)

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

# Simulated word embeddings for 4 words in each sentence, with 3-dimensional embeddings
# In practice, these could be 300-dimensional GloVe or Word2Vec embeddings
target_sentence = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.3, 0.2],
                            [0.5, 0.6, 0.7],
                            [0.8, 0.9, 0.7]])

generated_sentence = np.array([[0.1, 0.2, 0.3],
                               [0.4, 0.3, 0.2],
                               [0.5, 0.6, 0.7],
                               [0.8, 0.9, 0.7]])

# Compute the cost matrix: Euclidean distance between each pair of word embeddings
cost_matrix = np.sum((target_sentence[:, np.newaxis, :] - generated_sentence[np.newaxis, :, :]) ** 2, axis=2)

# Use the Hungarian algorithm to find the optimal assignment of target words to generated words
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Earth Mover's Distance (EMD) or Optimal Transport (OT) distance is the sum of the costs of these optimal assignments
emd = cost_matrix[row_ind, col_ind].sum()

print("Optimal Assignment Indices:", row_ind, col_ind)
print("Optimal Transport Distance:", emd)

```
In this example, the target_sentence and generated_sentence arrays simulate the word embeddings of words in a target and a generated sentence. The word embeddings are 3-dimensional for simplicity.

- The cost matrix is calculated as the Euclidean distance between each pair of word embeddings.
- The Hungarian algorithm (`linear_sum_assignment`) finds the optimal assignment that minimizes the total cost (or distance).
- The Earth Mover's Distance (EMD) or Optimal Transport (OT) distance is then computed as the sum of the costs of these optimal assignments.
- The output `row_ind` and `col_ind` gives the indices of the optimal assignments, and `emd` provides the optimal transport distance, which in this case is zero since both sentences are identical.


### 2. Equivalence of NLL, Cross-Entropy, and K-L Divergence in Classification Problems

 **Unveiling the Identity of NLL and Cross-Entropy**: Negative Log-Likelihood (NLL) and Cross-Entropy serve as cornerstone loss functions in machine learning, particularly in classification tasks and Natural Language Generation (NLG) models. These two loss functions are not just similar; they are mathematically identical when it comes to classification problems. This identity often goes unnoticed due to the different contexts in which each is typically introduced—Cross-Entropy often appears in binary classification scenarios, while NLL is more universally applied. Both aim to optimize the same objective function and reach their minimum or maximum values at identical points. For an in-depth exploration and practical demonstration, consult the Jupyter Notebook: [Comparison of NLL and Cross-Entropy](../src/Comparaision_NLL_Enthropy.ipynb).

 **Connection to Kullback-Leibler Divergence**: Intriguingly, when the ground truth labels are binary (either 0 or 1), the NLL loss function becomes a specific instance of Kullback-Leibler (K-L) Divergence. This is a widely-used metric to quantify the divergence between two probability distributions. In the context of binary classification, NLL, Cross-Entropy, and K-L Divergence essentially measure the same "distance" between the predicted and actual data distributions. For further insights, you may refer to this enlightening [article](http://www.awebb.info/probability/2017/05/18/cross-entropy-and-log-likelihood.html).

### 3. Computational Advantages and Optimization Techniques

#### Computational Efficiency of NLL
Negative Log-Likelihood (NLL) is often favored over direct likelihood calculations for computational reasons. Calculating the likelihood involves taking the product of probabilities, which can result in numerical underflow issues, especially when dealing with a large number of samples. NLL, on the other hand, works with the sum of log-probabilities, making it computationally more stable and efficient. This is particularly beneficial in high-dimensional spaces and large datasets, common scenarios in machine learning applications.

#### Optimization Using Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is a widely-used optimization algorithm for minimizing NLL. The algorithm updates the model parameters iteratively, using a subset of the data, making it more computationally efficient than batch gradient descent. This is especially useful for large-scale problems where using the entire dataset for each update is computationally expensive. The stochastic nature of the algorithm also helps escape local minima, providing a more robust optimization process.


### 4. Conclusion 

In the field of Natural Language Generation (NLG) and broader machine learning contexts, the importance of loss functions cannot be overstated. This discussion has been centered on two critical loss functions: Negative Log-Likelihood (NLL) and Optimal Transport (OT).

The NLL is a stalwart in classification models, serving as a practical avenue for implementing Maximum Likelihood Estimation (MLE). A notable point of confusion often arises when distinguishing NLL from Cross-Entropy. Despite different notations and contexts in which they are introduced, they are mathematically identical. This identity is especially significant since it bridges the gap between different notational systems and contexts, making it easier to translate insights across domains.

Moreover, when dealing with binary ground truth, NLL can be interpreted as a specific instance of Kullback-Leibler (KL) divergence. This adds another layer of understanding, as KL divergence is a measure of how one probability distribution diverges from a reference distribution. It provides a unifying thread between several important concepts in information theory and statistics, further emphasizing the ubiquity and utility of NLL as a loss function.

Optimal Transport (OT), on the other hand, has been gaining attention as an alternative loss function. Unlike NLL, OT focuses on finding optimal matchings between entire sets of weighted elements, making it useful for tasks requiring a holistic approach to comparing predicted and ground-truth distributions.

Each of these loss functions—NLL and OT—has unique advantages and specific applications. NLL is grounded in statistical theory and is computationally more tractable. OT offers a geometrically-inspired perspective on the space of distributions and is particularly useful when dealing with structured or distributional data. 

In summary, grasping these loss functions is more than just an intellectual endeavor; it's a critical step for anyone aiming to choose the most suitable optimization criterion for their specific problem, be it in NLG or other machine learning tasks. The mathematical interconnections between NLL, Cross-Entropy, and Kullback-Leibler divergence enrich our understanding of these optimization criteria, linking them in a web of mathematical equivalence and practical utility.

For more in-depth understanding, the articles [Cross Entropy and Log-Likelihood](http://www.awebb.info/probability/2017/05/18/cross-entropy-and-log-likelihood.html) and [Optimal Transport for Text Generation](https://arxiv.org/abs/1901.06283) are highly recommended readings.



#### Academic References

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
