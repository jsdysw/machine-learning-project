# machine learning project

## 01. Python tutorial and LaTex practice

* Learn usage of basic python syntax, numpy and matplotlib

## 02. Logistic regression for binary classification

* Classify two hand digit images into 0 and 1.

<img width="500" alt="data" src="https://user-images.githubusercontent.com/76895949/153762708-363c0baa-a9c3-4a32-a95d-a98240115cb7.png">
<img width="400" alt="pred1" src="https://user-images.githubusercontent.com/76895949/153762985-c896f02c-604b-4ce3-94d9-a84afec1b674.png"><img width="400" alt="pred2" src="https://user-images.githubusercontent.com/76895949/153762989-56219c75-d9d8-4e04-904c-058e3db64503.png">

## 03. Logistic regression for binary classification

* Classify multiple hand digit images of 0 and 1.

<img width="400" alt="data0" src="https://user-images.githubusercontent.com/76895949/153762787-913e2c46-c632-426a-94ab-9c345a247745.png"><img width="400" alt="data1" src="https://user-images.githubusercontent.com/76895949/153762790-66ac17b7-0361-4ffa-86e1-ed99c61bdf06.png">

<img width="450" alt="pred0" src="https://user-images.githubusercontent.com/76895949/153762793-8c902418-3027-46b2-85d8-2fe4d8f08897.png"><img width="450" alt="pred1" src="https://user-images.githubusercontent.com/76895949/153762795-1aae7b66-0f87-45ca-b0ab-bf8654981dec.png">

* testing accuracy at iteration 900 : 1.0000000000

## 04. Classification for multiple classes based on softmax and cross entropy

* Logistic regression for multi-class classification.
* Classify multiple hand digit images of 0, 1, 2, 3 and 4.
* Use One-Hot Encoding.

<img width="596" alt="accuracy" src="https://user-images.githubusercontent.com/76895949/153763034-c4863329-62ed-4b6d-a909-39a4a1b6f740.png">
![data0](https://user-images.githubusercontent.com/76895949/153763038-a35d4e63-1f7a-4cb3-a2a0-652f5ac7bd31.png)
![data1](https://user-images.githubusercontent.com/76895949/153763039-4adea02c-42e8-4edb-8fb9-07f5bc4100bf.png)



testing accuracy at iteration 900 : 0.9733333333

## 05. Classification for multiple classes with bias, weight-decay and stochastic gradient descent

* Logistic regression for multi-class classification.
* Classify multiple hand digit images of 0 ~ 9.
* Use One-Hot Encoding.

![accuracy-minibatch](https://user-images.githubusercontent.com/76895949/153763050-3239f975-1dec-4a7a-9618-acdde1fd49c0.png)
![accuracy-weightdecay](https://user-images.githubusercontent.com/76895949/153763052-a85b2e46-09a6-4c85-93aa-60655d94a661.png)
![data0](https://user-images.githubusercontent.com/76895949/153763055-fc936b82-0fde-45c4-a70c-b21a42d01424.png)
![data1](https://user-images.githubusercontent.com/76895949/153763057-5d083b41-0f50-4009-94a7-1fcb988500d8.png)
![data2](https://user-images.githubusercontent.com/76895949/153763058-bb58b2e9-a5a6-4614-9377-886d4e504768.png)
![data3](https://user-images.githubusercontent.com/76895949/153763060-fb1d236f-50d8-496e-b92a-ffb1144b36b7.png)

* testing accuracy (mean) at different mini-batch, weight-decay 0
* testing accuracy (mean) at different weight-decay, mini-batch0

## 06. Classification for multiple classes using Pytorch library

* Multi-class classification based on Softmax and Cross-Entropy using pytorch.
* Classify multiple hand digit images of 0 ~ 9.
* Use multiple layers for neural network.
* best testing (mean) accuracy within the last 10 epochs : 97.7875000000

![accuracy](https://user-images.githubusercontent.com/76895949/153763073-bd6ba547-bd21-4621-aea7-7f588c8be641.png)
![model](https://user-images.githubusercontent.com/76895949/153763075-c4db3193-fba6-42ca-a4db-ca964b855259.png)

