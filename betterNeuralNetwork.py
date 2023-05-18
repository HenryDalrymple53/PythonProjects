

import numpy as np
import pandas as pd
import random
#Potentially use leaky relu for weights and bias, and use sigmoid for output normalization.
#Avoiding vanishing gradient while keeping outputs scaled between -1 and 1
class NeuralNetwork:
    
    #Initialize weights and bias for dimensionality given
    def __init__(self, layerLengths):
        #Initializing variables
        self.weights = []
        self.bias = []
        self.answers = []
        self.input = []
        self.hlVals = []
        self.hlSig = []
        self.output = []
        self.DCoW = []
        self.DCoB = []
        self.sumCW = []
        self.sumCB = []
        self.only123 = []
        self.options = []
        #Length of output layer
        outputLength = 10
        #Used for measuring accuracy, and for specific number accuracy
        self.correct = 0
        self.correctList = [0,0,0,0,0,0,0,0,0,0]
        self.incorrectList = [0,0,0,0,0,0,0,0,0,0]
        #Reading in Data and shuffling it, no partition between test and train sets, just trying to get the system down first
        self.data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
        # Reads in all the data from the csv into an array
        self.data = np.array(self.data[0:len(self.data)]) 
        #Shuffles data
        random.shuffle(self.data) 
        
        #fills out an array of answers from the training set
        for x in self.data:
            self.answers.append(x[0]) 
            prev = 784
        for i in range(0,len(layerLengths)):
            self.weights.append((np.random.normal(size=(layerLengths[i], prev)) * np.sqrt(2/layerLengths[i])))
            prev = layerLengths[i]
            self.bias.append((np.random.normal(size=(prev, 1)) * np.sqrt(2/prev)))
        self.weights.append((np.random.normal(size=(10, prev)) * np.sqrt(2/10)))
        self.bias.append(np.random.normal(size=(10, 1)) * np.sqrt(2/10))
    
    def sigmoid(self, x): #Sigmoid squishes any number given to it
        sig = 1 / (1 + np.exp(-x))
        return sig
    def sigdir(self,x): #Sigmoid der
        return self.sigmoid(x) * (1- self.sigmoid(x))

    def leaky_Relu(self,x): #Unused leaky_relu
        count = 0
        for val in x:
             x[count] = val*0.01 if val < 0 else val
             count+=1
        return x

    def  leaky_Relu_Derivative(self,x): # Unused leaky_relu derv
        count = 0
        for val in x:
            x[count] = 0.01 if val < 0 else 1
            count+=1
        return x
    def forwardProp(self):# Run through the application, starting at the input set, using matrix multiplication

        #Start at input, multiply through weights, end at output. Similar structure to top
        self.hlVals.append(self.weights[0].dot(self.input) + self.bias[0])
        self.hlSig.append(self.sigmoid(self.hlVals[0]))
        for i in range(1,len(self.weights)):
            self.hlVals.append(self.weights[i].dot(self.hlSig[i-1]) + self.bias[i])
            self.hlSig.append(self.sigmoid(self.hlVals[i]))
            

    def backProp(self, ind):#Backpropigate through the values, finding gradient descent curves for each
                            #weight and bias
        #Cost of the function
        prev = 2*(self.hlSig[len(self.hlSig)-1]-self.responseCleaner(self.answers[ind]))
        for i in range(len(self.hlSig)-1,0,-1):
            
            DAoZn = self.sigdir(prev)   
            DZoWn = np.reshape(self.hlSig[i-1],(1,len(self.hlSig[i-1])))
            DZnA1 = np.reshape(self.weights[i],(len(self.hlSig[i-1]),len(self.hlSig[i])))
            dot = DAoZn*prev
            self.DCoW.append(dot*DZoWn)
            self.DCoB.append(dot)
            prev = DZnA1.dot(dot)
        
        DAoZn = self.sigdir(prev)   #5,1
        DZoWn = np.reshape(self.input,(1,len(self.input))) #1,784
        DZnA1 = np.reshape(self.weights[0],(len(self.input),len(self.hlSig[0]))) #784,5
        self.DCoW.append((DAoZn*prev)*DZoWn)
        
        self.DCoB.append(DAoZn*prev)
        
    #Finds the most confident answer given, and creates a binary array 
    #With 1 at greatest value
    #TODO: Update with cleaner version, naive currently
    def responseCleaner(self, ans): 
        retArr = []
        for x in range(0,10):
            if(x==ans):
                retArr.append(1)
            else :
                retArr.append(0)
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        
        if(guess==ans):
            self.correctList[ans]+=1
            self.correct +=1
        else :
            self.incorrectList[ans]+=1
        return np.reshape(retArr,(len(retArr),1))

    def update(self, alpha): #This updates the values after the length of the batch.
        x = len(self.weights)-1
        for i in range(0,len(self.weights)):
            
            self.weights[i] -= np.multiply(self.sumCW[x],alpha)
            x-=1
        y = len(self.weights)-1
        for i in range(0,len(self.weights)):
            
            self.bias[i] -= np.multiply(self.sumCB[y],alpha)
            y-=1
        self.sumCB.clear()
        self.sumCW.clear()
        

    def train(self,iterations, alpha, epoch, batch): #Putting it all together
        data = []
       
        accuracy = []
        for x in range(0,epoch):
            print(f"Start Val: {self.only123[0:5]}")
            print(self.weights[len(self.weights)-1])
            for i in range(0,iterations):
                
                self.trainInput(i)
                self.forwardProp()
                
                self.backProp(i)
                #The batch collects the weights over a certain # of iterations, then applying them at the end
                #of the batch.
                if(i%batch==0):
                    self.sumCW = self.DCoW
                    self.sumCB = self.DCoB
                elif(i%batch==batch-1):
                    self.update(alpha)
                else:
                    self.summation(alpha)
                self.clear()
                #Checks the accuracy per 1000 
                if(i%1000==0):
                    accuracy.append(self.correct/1000)
                    print(self.correct/1000)
                    self.correct = 0
            print(f"Next Epoch ----------------------")
            print(self.weights[len(self.weights)-1])
        print(f"Correct Guesses: {self.correctList}")
        print(f"Incorrect Guesses: {self.incorrectList}")
        
    def updateInput(self, input): 
        self.input = input
    def clear(self):
        self.hlVals.clear()
        self.hlSig.clear()
        self.DCoW.clear()
        self.DCoB.clear()
    def summation(self, alpha): #Is called throughout the length of the batch to add all of the weights and biases
        
        for i in range(0,len(self.weights)):
            
            self.sumCW[i] += np.multiply(self.DCoW[i],alpha)
            
        
        for i in range(0,len(self.weights)):
            
            self.sumCB[i] += np.multiply(self.DCoB[i],alpha)
            
    def trainInput(self, ind): # Changes values of Mnist from 0-255 to 0-1
        
        self.input =  np.reshape(self.data[ind][1:785],(784,1))
        for i in range(0,len(self.input)):
            val = self.input[i]
            if(val>=230):
                self.input[i]=1
            else :
                self.input[i]=0
    def guessNum(self): #Meant to be used with camera information
        self.forwardProp()
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        print(guess)
