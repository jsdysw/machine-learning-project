# Logistic regression for a binary classification

## 1. Problem Definition

### (1) Data

- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- input data is given as the file [assignment_03_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/03/assignment_03_data.npz)
- both `training` and `testing` data have the same structure that consists of a pair of image $`x`$ and label $`y`$ where $`x`$ denotes a list of 2-dimensional matrices and $`y`$ denotes a list of scalars
- $`x[0, :, :]`$ represents the first image and $`x[1, :, :]`$ represents the second image
- $`y[0]`$ represents the label for $`x[0, :, :]`$ and $`y[1]`$ represents the label for $`x[1, :, :]`$

### (2) Neural Network

- neural network $`f_w(x)`$ consists of a linear layer followed by the `sigmoid` activation function 
- neural network $`f_w(x)`$ for input $`x`$ is defined by:
```math
f_w(x) = \sigma( w^T x ),
```
where $`w`$ denotes weights in the linear layer and $`\sigma`$ denotes sigmoid function defined by:
```math
\sigma(z) = \frac{1}{1 + \exp(-z)}
```
- output $`h = f_w(x)`$ of the neural network $`f_w(x)`$ for input $`x`$ is considered as prediction value for the class of input as follows:
```math
\begin{cases}
l(x) = 0 & \colon h < 0.5 \\
l(x) = 1 & \colon h \ge 0.5,
\end{cases}
```
where $`l(x)`$ denotes a label function that determines the class of $`x`$

### (3) Loss

- loss function is defined by:
```math
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
```
where $`\ell_i(w)`$ denotes loss for a pair of data $`x_i`$ and label $`y_i`$ as defined by:
```math
\ell_i(w) = - \left\{ y_i \log{(f_w(x_i))} + (1 - y_i) \log{(1 - f_w(x_i))} \right\}
```

### (4) Optimization by Gradient Descent

- gradient descent step is given as follows:
```math
w^{(t+1)} \coloneqq w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
```
where the gradient is defined by:
```math
\begin{align}
\nabla \ell_i(w) & = - y_i \frac{1}{f_w(x_i)} \frac{\partial f_w(x_i)}{\partial w} - (1 - y_i) \frac{1}{1 - f_w(x_i)} \frac{\partial (1 - f_w(x_i))}{\partial w}\\
& = \left( f_w(x_i) - y_i \right) x_i
\end{align}
```
where we have:
```math
\frac{d \, \sigma(z)}{d \, z} = \sigma(z) (1 - \sigma(z))
```

### (5) Training

- training aims to determine the model parameters of the neural network and its associated loss function is optimized using the training data

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Initialization

- initialize all the weights $`w`$ in the neural network $`f_w`$ with $`0.001`$

### (6) Hyper-parameters

- use $`0.1`$ for the learning rate $`\eta`$
- use $`1000`$ for the number of gradient descent iterations

## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_03.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/03/assignment_03.ipynb)

- submit the completed notebok `assignment_03.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_03.ipynb` to PDF file and submit the PDF file

### (3) GitHub history

- make `git commit` at least at the specified steps and make further commits in such a way that your coding steps are effectively demonstrated
- export the GitHub history page to PDF file and submit the PDF file
