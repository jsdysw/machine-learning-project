# Image Denoising using Pytorch library

## 1. Problem Definition

### (1) Data

- data is given by the file  [assignment_08_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/08/assignment_08_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- both `training` and `testing` data have the same structure that consists of a pair of clean image $`I`$ and its corresponding noisy image $`J`$
- images $`I(x, y)`$ and $`J(x, y)`$ are 2-dimensional matrices of the size $`64 \times 64`$ (gray-scale images)

### (2) Neural Network

- construct a neural network $`f_w`$ parameterized by model parameters $`w = (w_1, w_2, \cdots, w_m)`$ in the form of auto-encoder 
- typical auto-encoder architecture consists of a pair of  encoder $`e_u`$ and decoder $`d_v`$ where $`u`$ and $`v`$ are model parameters for $`e_u`$ and $`d_v`$
- output of auto-encoder is obtained by composition of encoder and decoder as follows:
    ```math
    f_w(\cdot) = d_v(e_u(\cdot))
    ```
- typical encoder consists of layers including convolution, pooling, activation, batch normalization
- typical decoder consists of layers including upsampling, convolution, activation, batch normalization
- typical activation function for the output of the auto-encoder network for denoising is Sigmoid
  
### (3) Loss

- let $`\{J_i\}_{i=1}^n`$ be a set of training noisy images
- objective function $`\mathcal{\ell_i}(w; J_i)`$ is defined in terms of model parameters $`w`$ for given noisy image $`y`$ as follows:
    ```math
    \mathcal{\ell_i} = \mathcal{\ell}(w; J_i) = \| J_i - \tilde{J}_i \|_2^2 + \beta \| \nabla \tilde{J}_i \|
    ```
    where $`\tilde{J}_i = f(J_i; w)`$ and total variation of image $`J(x, y)`$ is defined by:
    ```math
    \| \nabla J \| = \sum_{x} \sum_{y} \left( \left \vert \frac{\partial J(x, y)}{\partial x} \right \vert + \left \vert \frac{\partial J(x, y)}{\partial y} \right \vert \right)
    ```
- objective function $`\mathcal{L}(w)`$ for $`\{ J_i \}_{i=1}^n`$ is defined by:
  ```math
  \mathcal{L}(w) = \sum_{i=1}^n \ell_i + \alpha \| w \|_2^2 
  ```

### (4) Optimization by Stochastic Gradient Descent

- use stochastic gradient descent (sgd) optimizer in the pytorch library

### (5) Training

- training aims to determine the optimal model parameters $`w`$ of the neural network $`f_w`$ by optimizing its associated loss function using the training data

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Evaluation

- use PSNR for the evaluation of the performance
- $`PSNR`$ for $`(I, J, f_w)`$ is computed by:
  ```math
  PSNR = 10 * \log_{10} \left( \frac{MAX(I)^2}{MSE} \right)
  ```
  where $`MSE`$ is defined by:
  ```math
  MSE = \| f_w(J) - I \|_2^2
  ```

### (8) Initialization

- initialize all the weights in the neural network in order to achieve best performance (PSNR) based on the testing data
  
### (9) Hyper-parameters

#### select the followings in such a way that the best performance (PSNR) based on the testing data can be achieved

- neural network architecture
- initialization of the weights in the neural network
- number of epochs
- size of mini-batch
- learning-rate
- momentum
- weight-decay ($`\alpha`$)
- weight for total variation ($`\beta`$)
