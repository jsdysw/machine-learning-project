# Classification for multiple classes using Pytorch library

## 1. Problem Definition

### (0) Tutorial

- [PyTorch tutorials](https://pytorch.org/tutorials/)

### (1) Data

- input data consists of `training` data and `testing` data
- `training` data is used for training neural network
- `testing` data is used for validating the trained neural network
- input data is given as the file [assignment_06_data.npz](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/06/assignment_06_data.npz)
- both `training` and `testing` data have the same structure that consists of a pair of image $`x`$ and label $`y`$ where $`x`$ denotes a list of 2-dimensional matrices and $`y`$ denotes a list of scalars
- $`x[0, :, :]`$ represents the first image and $`x[1, :, :]`$ represents the second image
- $`y[0]`$ represents the label for $`x[0, :, :]`$ and $`y[1]`$ represents the label for $`x[1, :, :]`$
- $`x`$ represent digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- $`y`$ represent the labels for the digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- label is defined by one-hot encoding scheme as follows:
  - label for the digit 0 is defined by $`0`$
  - label for the digit 1 is defined by $`1`$
  - label for the digit 2 is defined by $`2`$
  - label for the digit 3 is defined by $`3`$
  - label for the digit 4 is defined by $`4`$
  - label for the digit 5 is defined by $`5`$
  - label for the digit 6 is defined by $`6`$
  - label for the digit 7 is defined by $`7`$
  - label for the digit 8 is defined by $`8`$
  - label for the digit 9 is defined by $`9`$

### (2) Neural Network

- construct a neural network using a series of convolutional layers and fully connected layers
- include bias for both the convolutional layers and the fully connected layers
- use ReLU for the activation function
- use maxpooling for down-sampling the feature
- the output of the neural network is the output of the final linear later (do not add Softmax at the final layer)

### (3) Loss

- use the softmax and the cross-entropy
- use CrossEntropyLoss in the pytorch library
- use a Linear layer for the output of the neural network since CrossEntropyLoss combines the softmax and the cross-entropy loss

### (4) Optimization by Stochastic Gradient Descent

- use stochastic gradient descent (sgd) optimizer in the pytorch library

### (5) Training

- training aims to determine the model parameters of the neural network and its associated loss function is optimized using the training data

### (6) Testing

- testing aims to validate the generality of the trained neural network using the testing data

### (7) Initialization

- initialize all the weights in the neural network in order to achieve best (mean) accuracy using the testing data
  
### (8) Hyper-parameters

#### select the followings in such a way that the best (mean) accuracy based on the testing data can be achieved
- neural network architecture
- initialization of the weights in the neural network
- number of epochs
- size of mini-batch
- learning-rate
- weight-decay
  
## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_06.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/06/assignment_06.ipynb)

- submit the completed notebok `assignment_06.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_06.ipynb` to PDF file and submit the PDF file

### (3) GitHub history

- make `git commit` at least 10 times in such a way that your coding steps are effectively demonstrated
- export the GitHub history page to PDF file and submit the PDF file

## 3. Evaluation

### based on the best testing (mean) accuracy

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
