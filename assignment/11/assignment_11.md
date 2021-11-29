# Image de-blurring by a supervised learning using Pytorch library

## 1. Problem Definition

### (1) Data

- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- input data is given as the file [assignment_11_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/11/assignment_11_data.npz)
- both `training` and `testing` data have the same structure that consists of a pair of clean image $`x`$ and its corrupted blurry image $`y`$
- images are 2-dimensional matrices of the size $`256 \times 256`$ (gray-scale images)
- data augmentation can be used for training

### (2) Neural Network

- construct a neural network in the form of auto-encoder that consists of encoder and decoder
- typical encoder consists of layers including convolution, pooling, batch normalization, activation
- typical decoder consists of layers including upsampling, convolution, batch normalization, activation
- typical activation function for the output of the network is Sigmoid
  
### (3) Loss

- use the mean squared error (the squared $`L_2`$-norm) between the prediction (de-blurred) and the ground truth (original)

### (4) Optimization

- use any optimization algorithms such as SGD, Adam, AdaGrad, RMSProp

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
  
## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_11.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/11/assignment_11.ipynb)

- submit the completed notebok `assignment_11.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_11.ipynb` to PDF file and submit the PDF file

### (3) GitHub history

- make `git commit` at least 10 times in such a way that your coding steps are effectively demonstrated
- export the GitHub history page to PDF file and submit the PDF file

## 3. Evaluation

### based on the best testing PSNR within the last 10 epochs

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
