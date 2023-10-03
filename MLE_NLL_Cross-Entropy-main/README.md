This notebook aims to give an understanding of the concepts of Maximum Likelihood Estimation (MLE), Negative Log-Likelihood (NLL), Cross Entropy, and Kullback-Leibler (K-L) divergence.

## Theoretical Foundations

- **Maximum Likelihood Estimation (MLE)**: Explain the principle of MLE and how it aims to find the parameter values that maximize the likelihood of making the observed data.
- **Negative Log-Likelihood (NLL)**: Discuss why NLL is used as a loss function, emphasizing its relationship with MLE. Mention that NLL is essentially the negative logarithm of the likelihood function.
- **Cross Entropy**: Introduce the concept of entropy and then cross-entropy. Explain how cross-entropy measures the dissimilarity between the true distribution \( p \) and the predicted distribution \( q \).
- **Kullback-Leibler (K-L) Divergence**: Describe K-L divergence as a measure of how one probability distribution diverges from a second, expected probability distribution.

## Mathematical Equations
- Show the mathematical equations for MLE, NLL, Cross Entropy, and K-L Divergence.
- Explain the mathematical equivalence between NLL and Cross Entropy, emphasizing that they are minimized/maximized at the same points.

## Practical Applications
- Discuss how these concepts are used in machine learning, particularly in classification problems and Natural Language Generation (NLG) models.
- Explain the role of softmax activation in converting raw scores to probabilities in neural networks.

## Clarifications and Misconceptions
- Address common misconceptions, such as the difference between per-example loss and batch loss.
- Explain why the K-L divergence is not symmetric, i.e., \( D_{KL}(p||q) \neq D_{KL}(q||p) \).

## Computational Aspects
- Discuss the computational benefits of using NLL and why it's preferred over calculating the likelihood directly.
- Mention stochastic gradient descent as a common optimization algorithm used to minimize NLL.

## Empirical Evidence
- Present empirical results or case studies that demonstrate the effectiveness of using these loss functions in real-world applications.

## Academic Papers
I found some academic papers that could help deepen your understanding of the relationship between Negative Log-Likelihood, Maximum Likelihood Estimation, Cross Entropy, and Kullback-Leibler Divergence:

### [The Estimation of Distributions and the Minimum Relative Entropy Principle](https://dx.doi.org/10.1162/1063656053583469)
- **Authors**: H. Mühlenbein, Robin Höns
- **Abstract**: This paper discusses the relationship of Estimation of Distribution Algorithms (EDA) to algorithms in statistics and artificial intelligence. It focuses on the role of maximum entropy approximations and Kullback-Leibler divergence.
- **Cited by**: 97
- [PDF](http://www.muehlenbein.org/minrel.PDF)

### [Adaptation in log-concave density estimation](https://dx.doi.org/10.17863/CAM.11980)
- **Authors**: Arlene K. H. Kim, Adityanand Guntuboyina, R. Samworth
- **Abstract**: The paper discusses the log-concave maximum likelihood estimator of a density and its rate of convergence with respect to various global loss functions, including Kullback–Leibler divergence.
- **Cited by**: 45
- [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-46/issue-5/Adaptation-in-log-concave-density-estimation/10.1214/17-AOS1619.pdf)

### [Entropy and Divergence Associated with Power Function and the Statistical Application](https://dx.doi.org/10.3390/e12020262)
- **Authors**: S. Eguchi, Shogo Kato
- **Abstract**: This paper discusses the relationship between entropy and the maximum likelihood method, focusing on Kullback-Leibler divergence.
- **Cited by**: 44
- [PDF](https://www.mdpi.com/1099-4300/12/2/262/pdf?version=1424784678)

### [Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution](https://dx.doi.org/10.1109/TIFS.2021.3092050)
- **Authors**: Akira Ito, Kotaro Saito, Rei Ueno, N. Homma
- **Abstract**: This paper discusses the problems caused by data imbalance in deep learning-based attacks and introduces Kullback–Leibler divergence as a metric to measure this effect.
- **Cited by**: 18
- [PDF](https://ieeexplore.ieee.org/ielx7/10206/9151439/09464254.pdf)

Would you like to go into more detail on any of these papers? Also, would you like to save any of these citations to your Zotero reference manager?
