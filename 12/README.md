# Image Generation via Generative Adversarial Networks

## 1. Problem Definition

### (1) Data

- input data is given as the file [assignment_12_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/12/assignment_12_data.npz)
- input data consist of binary images of square shapes with varying sizes and locations
- input data are 2-dimensional matrices of the size $`32 \times 32`$ (gray-scale images)

### (2) Neural Network

- construct a neural network for the generator and the discriminator
- discriminator is a neural network for classifying input image as real or fake
- generator is a neural network for mapping a latent vector to a fake image
  
### (3) Loss

- min-max loss: 
  ```math
  \mathbb{E}_x \left[ \log{(D(x))}\right] + \mathbb{E}_z \left[ \log{(1 - D(G(z)))}\right]
  ```
- The generator $`G`$ tries to minimize this function while the discriminator $`D`$ tries to maximize it.

### (4) Optimization

- use any optimization algorithms such as SGD, Adam, AdaGrad, RMSProp

### (5) Training

- the discriminator and the generator are updated in an alternative way

### (6) Testing

- take a set of latent vectors and apply them to the trained generator

### (7) Evaluation

- use the IoU between the generated images and their corresponding square shapes

### (8) Initialization

- initialize all the weights in the neural network in order to achieve best performance
  
### (9) Hyper-parameters

#### select the followings in such a way that the best performance can be achieved

- neural network architecture
- initialization of the weights in the neural network
- number of epochs
- size of mini-batch
- learning-rate
