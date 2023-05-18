#Framework for a single hidden layer NN, with variable neurons in hidden layer. 
import numpy as np
import cv2
class NeuralNetwork:
    inputLayer = []
    weights = []
    hiddenLayer = []
    neuronOutput = []
    output = []
    def __init__(self, inputLayer,weights,hiddenLayer,neuronOutput):
        self.inputLayer = inputLayer
        self.weights = weights
        self.hiddenLayer = hiddenLayer
        self.neuronOutput = neuronOutput
    def run(inputLayer, weights, hiddenLayer, neuronOutput):
        ind = 0
        for i in inputLayer:
            sum = 0
            for w in weights:
                sum = sum + (i*w)
                sm = 1/(1 + np.exp(-sum))
            hiddenLayer[ind] = sum
            ind = ind + 1
        


    
    



