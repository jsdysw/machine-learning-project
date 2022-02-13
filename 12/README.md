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
  
## 2. Submission

### (1) notebook file for the python codes and the results

- use python3 programming to compelete the notebook [assignment_12.ipynb](https://gitlab.com/cau-class/neural-network/2021-2/assignment/-/blob/main/12/assignment_12.ipynb)

- submit the completed notebok `assignment_12.ipynb`

### (2) notebook file in PDF format

- export the completed notebook `assignment_12.ipynb` to PDF file and submit the PDF file

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
