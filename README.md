# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and Load the Dataset

2.Drop Irrelevant Columns (sl_no, salary)

3.Convert Categorical Columns to Category Data Type

4.Encode Categorical Columns as Numeric Codes

5.Split Dataset into Features (X) and Target (Y)

6.Initialize Model Parameters (theta) Randomly

7.Define Sigmoid Activation Function

8.Define Logistic Loss Function (Binary Cross-Entropy)

9.Implement Gradient Descent to Minimize Loss

10.Train the Model by Updating theta Iteratively

11.Define Prediction Function Using Threshold (0.5)

12.Predict Outcomes for Training Set

13.Calculate and Display Accuracy

14.Make Predictions on New Data Samples
## Program // Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```
```
print(y_pred)
```
```
print(Y)
```
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
print("Name:Hashwatha M")
print("Reg No:212223240051")
```
## Output :
## Dataset
![image](https://github.com/user-attachments/assets/bfaab922-38b6-4c0b-b1c2-25e3269e3509)

## dtypes
![image](https://github.com/user-attachments/assets/0d8b1bce-3043-4646-8a76-f185cffd268f)

## dataset
![image](https://github.com/user-attachments/assets/21ee6cf9-d70c-4dda-a5cb-268822a4d854)

## y array
![image](https://github.com/user-attachments/assets/6b6fece9-9129-4540-a521-78725c1632d7)

## Accuracy
![image](https://github.com/user-attachments/assets/f526d449-e2f6-4635-a247-2503e1795858)

## y
![image](https://github.com/user-attachments/assets/b7c33933-7841-4719-88ea-54022cdf6f73)

## y_prednew
![Screenshot 2025-04-28 162215](https://github.com/user-attachments/assets/37b6cdc6-eb81-40a0-9867-d87fa13a5b3a)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
