# Image Segmentation in an unsupervised learning framework using Pytorch library

## 1. Problem Definition

### (1) Data

- data is given by the file  [assignment_10_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/10/assignment_10_data.npz)
- data consists of binary images for squares with varying sizes
- dataset is constructed with original binary images and their noisy images with a given noise standard deviation
- data is split into `training` dataset and `testing` dataset
- both `training` and `testing` datasets have the same structure that consists of a pair of binary images for squares and their noisy images with a given noise level
- images are 2-dimensional matrices of the size $`32 \times 32`$
- data augmentation can be used for training

### (2) Neural Network

- construct a neural network $`f_w`$ parameterized by model parameters $`w = (w_1, w_2, \cdots, w_m)`$ in the form of auto-encoder
- typical auto-encoder architecture consists of a pair of  encoder $`e_u`$ and decoder $`d_v`$ where $`u`$ and $`v`$ are model parameters for $`e_u`$ and $`d_v`$
- output of auto-encoder is obtained by composition of encoder and decoder as follows:
  ```math
  f_w(\cdot) = d_v(e_u(\cdot))
  ```
- typical encoder consists of layers including convolution, pooling, batch normalization, activation
- typical decoder consists of layers including upsampling, convolution, batch normalization, activation
- typical activation function for the output of the auto-encoder network for segmentation is `Sigmoid`
  
### (3) Loss

- let $`\{I_i\}_{i=1}^n`$ be a set of training images
- let $`\phi_i = f_w(I_i)`$ be output of neural network $`f_w`$ for input $`I_i`$
- objective function $`\ell_i(w; I_i)`$ is defined in terms of model parameters $`w`$ for input $`I_i`$ and output $`\phi_i`$ as follows:
  ```math
  \ell_i(w) = \ell(w; I_i, \phi_i) = \int_{\Omega} \phi_i(x) \, (I_i(x) - a_i)^2 dx + \int_{\Omega} (1 - \phi_i(x)) \, (I_i(x) - b_i)^2 dx + \lambda \| \nabla \phi_i \|
  ```
  where estimates $`a_i \in \mathbb{R}`$ and $`b_i \in \mathbb{R}`$ for inside and outside of segmenting regions represented by $`\phi_i`$, respectively
- estimates $`a_i`$ and $`b_i`$ are obtained by:
  ```math
  a_i = \frac{\int_{\Omega} I_i(x) \phi_i(x) dx}{\int_{\Omega} \phi_i(x) dx}, \qquad b_i = \frac{\int_{\Omega} I_i(x) (1 - \phi_i(x)) dx}{\int_{\Omega} (1 - \phi_i(x)) dx}
  ```
  and Total Variation of $`\phi_i`$ is defined by:
  ```math
  \| \nabla \phi_i \| = \int_{\Omega} \left\vert \frac{\partial \phi_i(x)}{\partial x_1} \right\vert + \left\vert \frac{\partial \phi_i(x)}{\partial x_2} \right\vert dx
  ```
  where $`x = (x_1, x_2)`$
- objective function $`\mathcal{L}(w)`$ for training set $`\{I_i\}_{i=1}^n`$ is defined by:
  ```math
  \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n \ell_i(w) + \alpha \| w \|_2^2
  ```

### (4) Optimization by Stochastic Gradient Descent

- use stochastic gradient descent (sgd) optimizer in the pytorch library

### (5) Training

- training aims to determine the optimal model parameters $`w`$ of the neural network $`f_w`$ by optimizing its associated loss function using the training data
- you can use data augmentation

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Evaluation

- segmentation result for input image $`I`$ is obtained in the form of binary image by taking a threshold $`0.5`$ with respect to the inference $`\phi = f_w(I)`$
- use Intersection over Union (IoU) for the evaluation of the performance
- `IoU` is computed by:
  ```math
  \textrm{IoU} = \frac{\textrm{Area of Intersection}}{\textrm{Area of Union}}
  ```
  
### (8) Initialization

- initialize all the weights in the neural network in order to achieve best performance based on the testing data
  
### (9) Hyper-parameters

#### select the followings in such a way that the best performance based on the testing data can be achieved

- neural network architecture
- initialization of the weights in the neural network
- number of epochs
- size of mini-batch
- learning-rate
- weight-decay ($`\alpha`$)
- regularization weight ($`\lambda`$)
  
## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_10.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/10/assignment_10.ipynb)

- submit the completed notebok `assignment_10.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_10.ipynb` to PDF file and submit the PDF file
- you can first convert jupyter `notebook` to `HTML` and then convert `HTML` to `PDF`

### (3) GitHub history

- make `git commit` at least 10 times in such a way that your coding steps are effectively demonstrated
- export the GitHub history page to PDF file and submit the PDF file

## 3. Evaluation

### based on the best testing accuracy within the last 10 epochs

- rank 01 : 10
- rank 02 - 03 : 9
- rank 04 - 06 : 8
- rank 07 - 10 : 7
- rank 11 - 15 : 6
- rank 16 - 21 : 5
- rank 22 - 28 : 4
- rank 29 - 36 : 3
- rank 37 - 45 : 2
- rank 46 - : 1
- wrong : 0
