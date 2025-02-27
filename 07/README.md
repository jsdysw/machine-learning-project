# Image Denoising using Pytorch library

## 1. Problem Definition

### (1) Data

- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- input data is given as the file [assignment_07_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/07/assignment_07_data.npz)
- both `training` and `testing` data have the same structure that consists of a pair of clean image $`x`$ and its corresponding noisy image $`y`$
- images are 2-dimensional matrices of the size $`64 \times 64`$ (gray-scale images)

### (2) Neural Network

- construct a neural network in the form of auto-encoder that consists of encoder and decoder
- typical encoder consists of layers including convolution, pooling, activation, batch normalization
- typical decoder consists of layers including upsampling, activation, batch normalization
- typical activation function for the output of the network is Sigmoid
  
### (3) Loss

- use the mean squared error (the squared $`L_2`$-norm) between the prediction (denoised) and the ground truth (clean)

### (4) Optimization by Stochastic Gradient Descent

- use stochastic gradient descent (sgd) optimizer in the pytorch library

### (5) Training

- training aims to determine the model parameters of the neural network and its associated loss function is optimized using the training data

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Evaluation

- use PSNR for the evaluation of the performance
- PSNR is computed by $`PSNR = 10 * \log_{10} \left( \frac{MAX(IMAGE)^2}{MSE} \right)`$

### (8) Initialization

- initialize all the weights in the neural network in order to achieve best performance (PSNR) based on the testing data
  
### (9) Hyper-parameters

#### select the followings in such a way that the best performance (PSNR) based on the testing data can be achieved

- neural network architecture
- initialization of the weights in the neural network
- number of epochs
- size of mini-batch
- learning-rate
- weight-decay
- momentum
