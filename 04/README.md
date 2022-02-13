# Classification for multiple classes with softmax and cross entropy

## 1. Problem Definition

### (1) Data

- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- input data is given as the file [assignment_04_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/04/assignment_04_data.npz)
- both `training` and `testing` data have the same structure that consists of a pair of image $`x`$ and label $`y`$ where $`x`$ denotes a list of 2-dimensional matrices and $`y`$ denotes a list of vectors
- $`x[0, :, :]`$ represents the first image and $`x[1, :, :]`$ represents the second image
- $`y[0]`$ represents the label for $`x[0, :, :]`$ and $`y[1]`$ represents the label for $`x[1, :, :]`$
- $`x`$ represent digits 0, 1, 2, 3, 4
- $`y`$ represent the labels for the digits 0, 1, 2, 3, 4
- label is defined by one-hot encoding scheme as follows:
  - label for the digit 0 is defined by $`(1, 0, 0, 0, 0)`$
  - label for the digit 1 is defined by $`(0, 1, 0, 0, 0)`$
  - label for the digit 2 is defined by $`(0, 0, 1, 0, 0)`$
  - label for the digit 3 is defined by $`(0, 0, 0, 1, 0)`$
  - label for the digit 4 is defined by $`(0, 0, 0, 0, 1)`$

### (2) Neural Network

- neural network $`f_w(x)`$ consists of a linear layer followed by the `softmax` activation function 
- neural network $`f_w(x)`$ for input $`x`$ is defined by:
```math
f_w(x) = \sigma( w^T x ),
```
where $`w`$ denotes weights in the linear layer and $`\sigma`$ denotes softmax (normalized exponential) function defined by:
```math
\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
```
where $`z = (z_1, z_2, \cdots, z_K) \in \mathbb{R}^K`$ and $`z_i = w_i^T x`$
- output $`h = f_w(x)`$ of the neural network $`f_w(x)`$ for input $`x`$ is considered as prediction value $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ where $`h_i = \sigma(z)_i`$ for each class
- label of input $`x`$ is determined by:
```math
l(x) = \arg\max_j \sigma(z)_j
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

### (3) Loss

- loss function is defined by the cross entropy between $`h = (h_1, h_2, \cdots, h_K) \in \mathbb{R}^K`$ and $`y = (y_1, y_2, \cdots, y_K) \in \mathbb{R}^K`$:
```math
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
```
where $`\ell_i(w)`$ denotes loss for a pair of data $`x_i`$ and label $`y_i`$ as defined by:
```math
\ell_i(w) = - \sum_{j = 1}^K y_{i, j} \log(h_{i, j})
```

### (4) Optimization by Gradient Descent

- gradient descent step is given as follows:
```math
w^{(t+1)} \coloneqq w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
```
where the gradient is defined by:
```math
\begin{align}
\frac{\partial \ell}{\partial w_k} & = - y_k \frac{\log(h_k)}{\partial w_k} - \sum_{j \neq k} y_j \frac{\log(h_j)}{\partial w_k} \\
\frac{\log(h_k)}{\partial w_k} & =  h_k ( 1 - h_k ) x\\
\frac{\log(h_j)}{\partial w_k}  & = - (h_j \, h_k) x
\end{align}
```
and we have:
```math
\frac{\partial \ell}{\partial w_k} = (h_k - y_k) x
```

### (5) Training

- training aims to determine the model parameters of the neural network and its associated loss function is optimized using the training data

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Initialization

- initialize all the weights $`w`$ in the neural network $`f_w`$ with $`0.001`$

### (6) Hyper-parameters

- use $`0.5`$ for the learning rate $`\eta`$
- use $`2000`$ for the number of gradient descent iterations