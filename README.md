# machine learning project

## 01. Python tutorial and LaTex practice

* Learn usage of basic python syntax, numpy and matplotlib

## 02. Logistic regression for binary classification

* Classify two hand digit images into 0 and 1.

<img width="200" alt="data" src="https://user-images.githubusercontent.com/76895949/153762708-363c0baa-a9c3-4a32-a95d-a98240115cb7.png">

<img width="400" alt="pred1" src="https://user-images.githubusercontent.com/76895949/153762985-c896f02c-604b-4ce3-94d9-a84afec1b674.png"><img width="400" alt="pred2" src="https://user-images.githubusercontent.com/76895949/153762989-56219c75-d9d8-4e04-904c-058e3db64503.png">

## 03. Logistic regression for binary classification

* Classify multiple hand digit images of 0 and 1.
* **testing accuracy at iteration 900 : 1.0000000000**

<img width="200" alt="data0" src="https://user-images.githubusercontent.com/76895949/153762787-913e2c46-c632-426a-94ab-9c345a247745.png"><img width="200" alt="data1" src="https://user-images.githubusercontent.com/76895949/153762790-66ac17b7-0361-4ffa-86e1-ed99c61bdf06.png">

<img width="400" alt="pred0" src="https://user-images.githubusercontent.com/76895949/153762793-8c902418-3027-46b2-85d8-2fe4d8f08897.png"><img width="400" alt="pred1" src="https://user-images.githubusercontent.com/76895949/153762795-1aae7b66-0f87-45ca-b0ab-bf8654981dec.png">

## 04. Classification for multiple classes based on softmax and cross entropy

* Logistic regression for multi-class classification.
* Classify multiple hand digit images of 0, 1, 2, 3 and 4.
* Use One-Hot Encoding.
* **testing accuracy at iteration 900 : 0.9733333333**

<img width="200" alt="data0" src="https://user-images.githubusercontent.com/76895949/153763151-8a7ad5da-c5eb-4319-9804-6c846d6ea9e4.png"><img width="200" alt="data1" src="https://user-images.githubusercontent.com/76895949/153763160-b3a7394d-cbe3-4feb-b9f7-27277805bba6.png"><img width="500" alt="accuracy" src="https://user-images.githubusercontent.com/76895949/153763034-c4863329-62ed-4b6d-a909-39a4a1b6f740.png">

## 05. Classification for multiple classes with bias, weight-decay and stochastic gradient descent

* Logistic regression for multi-class classification.
* Classify multiple hand digit images of 0 ~ 9.
* Use One-Hot Encoding.
* **testing accuracy (mean) at different mini-batch, weight-decay 0**
* **testing accuracy (mean) at different weight-decay, mini-batch0**

<img width="200" alt="data0" src="https://user-images.githubusercontent.com/76895949/153763055-fc936b82-0fde-45c4-a70c-b21a42d01424.png"><img width="200" alt="data1" src="https://user-images.githubusercontent.com/76895949/153763057-5d083b41-0f50-4009-94a7-1fcb988500d8.png"><img width="200" alt="data2" src="https://user-images.githubusercontent.com/76895949/153763058-bb58b2e9-a5a6-4614-9377-886d4e504768.png"><img width="200" alt="data3" src="https://user-images.githubusercontent.com/76895949/153763060-fb1d236f-50d8-496e-b92a-ffb1144b36b7.png">

<img width="400" alt="accuracy-minibatch" src="https://user-images.githubusercontent.com/76895949/153763050-3239f975-1dec-4a7a-9618-acdde1fd49c0.png"><img width="400" alt="accuracy-weightdecay" src="https://user-images.githubusercontent.com/76895949/153763052-a85b2e46-09a6-4c85-93aa-60655d94a661.png">


## 06. Classification for multiple classes using Pytorch library

* Multi-class classification based on Softmax and Cross-Entropy using pytorch.
* Classify multiple hand digit images of 0 ~ 9.
* Constructed a neural network using a series of convolutional layers.
* **best testing (mean) accuracy within the last 10 epochs : 97.7875000000**

<img width="450" alt="model" src="https://user-images.githubusercontent.com/76895949/153765568-1014851f-25f1-40ae-bf2b-3d3ecb016cc9.png">

## 07. Image Denoising in a superivsed learning framework using Pytorch library

* Denoise the noised images.
* Constructed a neural network in the form of auto-encoder that consists of encoder and decoder.
* **best testing PSNR (mean) within the last 10 epochs = 25.1513428898**

<img width="400" alt="스크린샷 2022-02-14 오전 2 30 22" src="https://user-images.githubusercontent.com/76895949/153767058-928daec9-fb79-4f93-a896-6ed6bb29c05b.png"><img width="400" alt="스크린샷 2022-02-14 오전 2 30 16" src="https://user-images.githubusercontent.com/76895949/153767050-8f2d08e7-d56b-47d5-9983-0947fe5142cf.png">

<img width="583" alt="스크린샷 2022-02-14 오전 2 30 54" src="https://user-images.githubusercontent.com/76895949/153767082-e6709ad5-1776-4ee1-b0c9-4943c3b2a981.png">

## 08. Image Denoising in an unsuperivsed learning framework using Pytorch library

* Denoise the noised images.
* Constructed a neural network in the form of auto-encoder that consists of encoder and decoder.
* **best testing PSNR (mean) within the last 10 epochs = 25.4595453543**

<img width="400" alt="스크린샷 2022-02-14 오전 2 26 40" src="https://user-images.githubusercontent.com/76895949/153766917-013ccf20-42bf-49d9-b63e-1c1ce1a68453.png"><img width="400" alt="스크린샷 2022-02-14 오전 2 26 34" src="https://user-images.githubusercontent.com/76895949/153766927-800a20a8-5072-4833-a7b5-b8bf9be8c6c0.png">

<img width="590" alt="스크린샷 2022-02-14 오전 2 25 56" src="https://user-images.githubusercontent.com/76895949/153766870-4f9ee508-ffc7-4a6d-992d-7d8043038c62.png">

## 09. Image Segmentation in a supervised learning framework using Pytorch library

* Get the clear boundaries of the cat images from the original.
* Constructed a neural network in the form of auto-encoder that consists of encoder and decoder.
* **best testing accuracy within the last 10 epochs = 74.2580732318**

<img width="400" alt="스크린샷 2022-02-14 오전 2 21 51" src="https://user-images.githubusercontent.com/76895949/153766735-5a128cdc-4bd7-42b8-913e-ba1cedbcecb3.png"><img width="400" alt="스크린샷 2022-02-14 오전 2 21 44" src="https://user-images.githubusercontent.com/76895949/153766740-a24b9d76-ec58-44c3-9d4a-fd4b1c9e605c.png">

<img width="586" alt="스크린샷 2022-02-14 오전 2 20 04" src="https://user-images.githubusercontent.com/76895949/153766746-d5f00511-1be4-4753-b6e1-95b64af0353a.png">

## 10. Image Segmentation in an unsupervised learning framework using Pytorch library

* Get the clear boundaries of the sqaure images from the original noised sqaure images.
* Constructed a neural network in the form of auto-encoder that consists of encoder and decoder.
* **best testing accuracy within the last 10 epochs = 97.8995329178**

<img width="400" alt="스크린샷 2022-02-14 오전 2 14 18" src="https://user-images.githubusercontent.com/76895949/153766529-03915d70-50a7-413a-bbab-2ab60bb95b63.png"><img width="400" alt="스크린샷 2022-02-14 오전 2 14 32" src="https://user-images.githubusercontent.com/76895949/153766536-5552bfee-85ef-4d69-816b-35edad4942c8.png">

<img width="599" alt="스크린샷 2022-02-14 오전 2 14 40" src="https://user-images.githubusercontent.com/76895949/153766548-9d701615-7061-4896-bd82-734f2b69047b.png">

## 11. Image de-blurring by a supervised learning using Pytorch library

* De-blur the blurred images.
* Constructed a neural network in the form of auto-encoder that consists of encoder and decoder.
* **best testing PSNR (mean) within the last 10 epochs = 24.1219267082**

<img width="400" alt="before" src="https://user-images.githubusercontent.com/76895949/153766146-228d3f78-4151-421d-a557-b46f5b4906f0.png"><img width="400" alt="after" src="https://user-images.githubusercontent.com/76895949/153766150-83fbbb27-1bf3-400e-9f6b-b1105507404b.png">

<img width="584" alt="스크린샷 2022-02-14 오전 2 07 05" src="https://user-images.githubusercontent.com/76895949/153766168-16faa7f1-e805-4883-8f30-238c5c68a3da.png">

## 12. Image Generation via Generative Adversarial Networks

* Create square images by learning from square images.
* Constructed neural networks of a generator and a discriminator.
* **best accuracy within the last 10 epochs = 96.0030833364**

<img width="651" alt="스크린샷 2022-02-14 오전 2 01 21" src="https://user-images.githubusercontent.com/76895949/153765984-03a85858-b018-4c81-9fa7-4912c8448872.png">
