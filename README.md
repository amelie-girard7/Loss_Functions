## Instruction 1

The first is called the 'negative log-likelihood' (NLL) and is the universal loss function used to train classification models (including NLG models).
The NLL is the practical way to achieve 'maximum likelihood estimation' (MLE), because the likelihood and the log-likelihood are maximised by the same argument.
In addition, many readers struggle to realise that it's identical to another loss function, the so-called 'cross entropy', simply because the latter is always introduced for the two-class (i.e., binary) case only, and with different notations. But they are exactly the same.
Curiously, given that the ground truth is always 0 or 1, the NLL is also an instance of the Kullback-Leibler divergence, a popular way to measure the difference between two distributions.
I have managed to find an article that talks about all this:


[Reference](http://www.awebb.info/probability/2017/05/18/cross-entropy-and-log-likelihood.html)


#### Summary:
The article aims to clarify the relationship between Cross Entropy, Negative Log-Likelihood (NLL), and Kullback-Leibler (K-L) Divergence, particularly in the context of neural networks used for multi-class classification. The article emphasizes that while NLL is commonly used as a loss function in machine learning models, it is essentially identical to Cross Entropy under certain conditions. Furthermore, the K-L divergence is discussed as a measure of the "distance" between two probability distributions.

#### Detailed Insights:

1. **Negative Log-Likelihood (NLL) and Cross Entropy**:  
   - *Detail*: In a neural network for multi-class classification, using a softmax activation in the final layer, the NLL serves as the cost function. The article reveals that this NLL is identical to Cross Entropy when one considers the predicted probabilities $( \hat{y} $) and the true labels $( y $) as discrete probability distributions.
   - *Example*: If you have a three-class problem, and for a given instance, the true label $( y $) is [1, 0, 0] (one-hot encoded), and the predicted probabilities $( \hat{y} $) are [0.7, 0.2, 0.1]. Both NLL and Cross Entropy will use these distributions to compute the loss, yielding the same value.

2. **Entropy and Optimal Encoding**:  
   - *Detail*: The article explains the concept of entropy as the "expected number of bits" required to encode a message optimally given a probability distribution.
   - *Example*: If you have a discrete random variable with three possible outcomes $( x_1, x_2, x_3 $) with probabilities $( p_1, p_2, p_3 $), the entropy will quantify the average number of yes/no questions needed to identify an outcome based on these probabilities.

3. **Kullback-Leibler (K-L) Divergence**:  
   - *Detail*: K-L Divergence measures how one probability distribution diverges or differs from a second, reference probability distribution. It's often seen as a distance measure between distributions.
   - *Example*: If your model predicts a probability distribution $( \hat{y} $) and the true distribution is $( y $), the K-L divergence will quantify the "distance" between $( \hat{y} $) and $( y $).

4. **Symmetry in Cross Entropy and K-L Divergence**:  
   - *Detail*: The article points out that Cross Entropy and K-L Divergence are not symmetric, meaning $( H(p, q) \neq H(q, p) $) and $( D_{KL}(p||q) \neq D_{KL}(q||p) $).
   - *Example*: In a classification problem, if $( y $) is the true distribution and $( \hat{y} $) is the predicted one, the Cross Entropy $( H(y, \hat{y}) $) will differ from $( H(\hat{y}, y) $).

5. **Batch vs. Per-Example Loss**:  
   - *Detail*: The article clarifies that while per-example NLL can be interpreted as Cross Entropy, the NLL of a batch of data is essentially a sum of Cross Entropies for each instance, each based on a different model distribution.
   - *Example*: In a mini-batch with 10 examples, the overall NLL is the sum of the NLLs for each of these 10 examples, which can be seen as the Cross Entropy summed over all instances.

The article serves as a comprehensive guide to understanding these often-confused concepts, providing mathematical and intuitive explanations that are critical for anyone working with neural networks and probabilistic models.


-------------------------------------------------------------------------------------

## Instruction 2
The other topic is called optimal transport (OT). It is a classic optimisation framework that is able to find optimal assignments (i.e., matching coefficients) between the elements of two weighted sets, A = {(a_1, w_1),...(a_N,w_N)} and B ={(b_1, w_1),...(b_M,w_M)} (note that they can be of different size, N and M). Optimal transport has also found some use in NLG as a loss function alternative to the NLL/MLE: it compares the predictions made by a model with the ground truth, and their OT distance is used as a loss function for training. 

[Reference ](https://arxiv.org/abs/1901.06283)

#### Summary:
The paper "Improving Sequence-to-Sequence Learning via Optimal Transport" marks a pivotal advancement in the realm of Seq2Seq learning. It introduces an Optimal Transport (OT)-based loss function, which serves as a more effective alternative to the conventional Negative Log-Likelihood (NLL) loss function. Through empirical validation, the paper shows that Seq2Seq models trained using the OT-based loss function outperform those trained using standard loss functions like NLL across a variety of benchmarks. Moreover, the authors tackle the computational challenges tied to OT, proposing algorithms that make it computationally feasible to incorporate OT into large neural network training procedures.

#### Expanded Insights:

1. **Sequence-Level Regularization**:  
   - *Detail*: In traditional Seq2Seq learning, the NLL loss function operates at a token-by-token level. This granular approach often fails to capture the broader semantic relationships between tokens in a sequence.
   - *Example*: Imagine translating a complex sentence from English to French. A token-level loss might correctly translate individual words but could rearrange them in a way that the sentence loses its original meaning. An OT-based loss would consider the entire sequence, preserving the sentence's overall semantic integrity.

2. **End-to-End Supervised Training**:  
   - *Detail*: The OT-based framework allows for end-to-end supervised training, bypassing the need for intermediate stages or specialized architectures.
   - *Example*: In reinforcement learning setups, one might need a separate policy network and a value network, requiring multi-stage training. The OT-based approach simplifies this by allowing a single-stage, end-to-end training process.

3. **Semantic Preservation**:  
   - *Detail*: The OT loss ensures that the generated and the target sequences are not just similar at a token level but also semantically coherent at a sequence level.
   - *Example*: In text summarization, the goal isn't just to have a grammatically correct summary but also one that captures the essence of the original text. The OT loss would strive to minimize the semantic "transport cost" between the generated summary and the original text.

4. **Comparative Advantage over Existing Methods**:  
   - *Detail*: Unlike Reinforcement Learning or adversarial methods, which can suffer from issues like mode-trapping and gradient-vanishing, the OT-based approach provides a more stable training landscape.
   - *Example*: In Generative Adversarial Networks (GANs), the generator and discriminator can enter a mode where they no longer provide useful gradients for learning—a problem known as mode collapse. OT-based methods avoid such pitfalls.

5. **Versatility Across Tasks**:  
   - *Detail*: The OT-based loss function can be generalized to a range of NLP tasks, including machine translation, text summarization, and even speech recognition.
   - *Example*: In machine translation tasks, the model could be trained to consider not just the lexical but also the contextual "distance" between the source and target languages, thereby producing translations that are both accurate and contextually appropriate.

6. **Computational Efficiency**:  
   - *Detail*: OT calculations are inherently computationally expensive. The paper mitigates this by proposing efficient algorithms tailored for this specific application.
   - *Example*: Implementing traditional OT algorithms in a neural network could slow down the training process significantly, making it impractical for large datasets. The paper's efficient algorithmic solutions make it feasible to use OT even in large-scale applications.

The paper uses an OT-based loss function in Seq2Seq learning and also backs its theoretical contributions with empirical evidence, thereby substantiating its applicability and effectiveness in real-world tasks. This work opens up exciting new avenues for research, especially in applications where capturing higher-order dependencies between sequences is paramount.
 
