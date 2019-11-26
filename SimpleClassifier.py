
import numpy as np
from matplotlib import pyplot as plt
#data input into list
data = [[6000, 330, 1],[100, 80, 0],[5870, 300, 1],[15, 20, 0], [200, 100, 1], [150, 110, 0] ,[1000, 200, 1],[180, 80, 1],[40, 50, 0]]

#define the sigmoid function, an essential function in the classifier
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#take derivative of sigmoid and define the derivative
def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Weights and bias
W1 = np.random.randn()
W2 = np.random.randn()
B = np.random.randn()

# parameters
LearningRate = 0.01
EPOCHS = 100000



for num in range(EPOCHS):
    
    #random point in data
    randpoint = data[np.random.randint(len(data))]
    z = ((randpoint[0] * W1) + (randpoint[1] * W2) + B)/10
    prediction = sigmoid(z)
    #making a prediction using sigmoid, weights and bias
    
    target = randpoint[2]
    DistCost = np.square(prediction - target)
    # distcost gives us how far the prediction was from target and allows it to improve
    
    dcost_dprediction = 2 * (prediction - target) # dx^2 = 2x
    dprediction = sigmoid_d(z) #throw into dsigmoid 
    #take derivative of distcost inrelation to prediction and take derivative of prediction 

    dw1_dz = randpoint[0] #take derivative of dz in relation to w1
    dw2_dz = randpoint[1] #take derivative of dz in relation to w2
    db_dz= 1 # take derivative of dz in relation to db which is simply 1


    DCOST_dw1 =  dw1_dz* dcost_dprediction * dprediction  #recalculate cost in relation to weights and biases
    DCOST_dw2 = dw2_dz* dcost_dprediction * dprediction 
    DCOST_b = db_dz*dcost_dprediction * dprediction 

    W1 = W1 - (LearningRate * DCOST_dw1) #update w1,w2 and b 
    W2 = W2 - (LearningRate * DCOST_dw2)
    B = B - (LearningRate * DCOST_b)
#also known as back propogation 

print(f"Training is done, the final values are:\nW1: {W1}\nW2: {W2}\nB: {B}")
plt.show()