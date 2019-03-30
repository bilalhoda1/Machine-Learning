#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
#reading data from excel using pandas
df = pd.read_excel('Data.xlsx',sheet_name=0)
x1 = np.array(df['Weight'].tolist())
x2 = np.array(df['Height'].tolist())
y = np.array(df['y'].tolist())
#randomly assigning weights and bias
w1=random.randint(1,5)
w2=random.randint(1,5)
b=random.random()
#initializing values
step=0.00001
#calculates the linear combination of inputs and weights
def forward(x1,x2,w1,w2,b):
    return (x1*w1+x2*w2)+b
#calculates loss
def loss(y_pred_val,y):
    loss=((y_pred_val-y)*(y_pred_val-y))
    return loss
#loop for epochs
for epoch in range(10000):
    dw1=0
    dw2=0
    db=0
    l_sum=0
    print("\tprogress:", epoch)
    #main loop for finding weights using gradient descent
    for x1_val, x2_val, y_val in zip(x1, x2, y):
        y_pred_val = forward(x1_val,x2_val,w1,w2,b)
        l=loss(y_pred_val,y_val)
        l_sum += l
        dw1+=2*x1_val*(y_pred_val-y_val)
        dw2+=2*x2_val*(y_pred_val-y_val)
        db+=2*(y_pred_val-y_val)
        
    dw1/=len(y)
    dw2/=len(y)
    db/=len(y)
    w1=w1-step*dw1
    w2=w2-step*dw2
    b=b-step*db
    l_sum=l_sum/len(y)
print("weights: ", w1, w2, b)
print("predict (after training)",  "1 hour")
print("cost",l_sum)
