"""
Testing for easiness of making perceptron in python :)
"""
from keras.model import Sequential
from keras.layer import Dense
import numpy
numpy.random.seed(7)
dataset=numpy.loadtxt("/home/vaishnav/MLP/Wine data/train.csv",delimiter=",")
n=input("Enter Number of Neurons\n")
X=dataset[:,:13]
Y=dataset[:,13]
model=Sequential()
model.add(Dense(14,input_dim=13,init="uniform",activation='sigmoid'))
model.add(Dense(n,init="uniform",activation="sigmoid"))
model.add(Dense(3,init="uniform",activation="sigmoid"))

