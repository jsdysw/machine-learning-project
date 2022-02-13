# Image Segmentation in a supervised learning framework using Pytorch library

## 1. Problem Definition

### (1) Data

- data is given by the file  [assignment_09_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/09/assignment_09_data.npz)
- data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- both `training` and `testing` data have the same structure that consists of a pair of image $`I`$ and its corresponding segmentation mask $`M`$
- image $`I(x, y)`$ and mask $`M(x, y)`$ are 2-dimensional matrices of the size $`128 \times 128`$
- images and masks are resized to $`64 \times 64`$ by bilinear and nearest interpolations for images and masks, respectively, both for training and testing
- data augmentation can be used for training
- image size $`64 \times 64`$ should be used for inference from both the training and testing dataset

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

- let $`\{(I_i, M_i)\}_{i=1}^n`$ be a set of images $`I_i`$ and segmentation masks $`M_i`$ pairs
- let $`\hat{I}_i = f_w(I_i)`$ be output of neural network $`f_w`$ for input $`I_i`$
- objective function $`\mathcal{\ell_i}(w; I_i, M_i)`$ is defined in terms of model parameters $`w`$ for given image $`I_i`$ and segmentation mask $`M_i`$ using the `mean squared error` as follows:
  ```math
  \mathcal{\ell_i} = \mathcal{\ell}(w; I_i, M_i) = \| M_i - \hat{I}_i \|_2^2
  ```
- objective function $`\mathcal{\ell_i}(w; I_i, M_i)`$ is defined in terms of model parameters $`w`$ for given image $`I_i`$ and segmentation mask $`M_i`$ using the `cross-entropy` as follows:
  ```math
  \mathcal{\ell_i} = \mathcal{\ell}(w; I_i, M_i) = - \sum_{x, y} M_i(x, y) \log{\hat{I}_i(x, y)}
  ```
- objective function $`\mathcal{L}(w)`$ for training set $`\{ (I_i, M_i) \}_{i=1}^n`$ is defined by:
  ```math
  \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n \ell_i + \alpha \| w \|_2^2
  ```

### (4) Optimization by Stochastic Gradient Descent

- use stochastic gradient descent (sgd) optimizer in the pytorch library

### (5) Training

- training aims to determine the optimal model parameters $`w`$ of the neural network $`f_w`$ by optimizing its associated loss function using the training data
- you can use data augmentation
- you can use image patches for training

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data
- you have to use image size $`64 \times 64`$ for inference even though image patches are used in the training

### (7) Evaluation

- segmentation result for input image $`I`$ is obtained in the form of binary image by taking a threshold $`0.5`$ with respect to the inference $`\hat{I} = f_w(I)`$
- use Intersection over Union (IoU) for the evaluation of the performance
- `IoU` for $`(I, J, f_w)`$ is computed by:
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
  
## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_09.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/09/assignment_09.ipynb)

- submit the completed notebok `assignment_09.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_09.ipynb` to PDF file and submit the PDF file
- you can first convert jupyter `notebook` to `HTML` and then convert `HTML` to `PDF`

### (3) GitHub history

- make `git commit` at least 10 times in such a way that your coding steps are effectively demonstrated
- export the GitHub history page to PDF file and submit the PDF file

## 3. Evaluation

### based on the best testing IoU within the last 10 epochs

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
