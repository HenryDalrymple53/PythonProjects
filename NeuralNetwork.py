
import numpy as np
from scipy.special import expit
import pandas as pd
#Potentially use leaky relu for weights and bias, and use sigmoid for output normalization.
#Avoiding vanishing gradient while keeping outputs scaled between -1 and 1
class NeuralNetwork:
    weights = []
    bias = []
    answers = []
    input = []
    hlVals = []
    hlSig = []
    output = []
    DCoW = []
    DCoB = []
    sumCW = []
    sumCB = []
    only123 = []
    options = []
    
    correct = 0
    correctList = [0,0,0,0,0,0,0,0,0,0]
    incorrectList = [0,0,0,0,0,0,0,0,0,0]
    
    data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
    data = np.array(data[1:])
    i = 0
    
    for x in data:
        answers.append(x[0])
        if(x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4):
            only123.append(i)
        i+=1
    hiddenLayerLength = 0
    def __init__(self, layerLengths):
        prev = 784
        
        for i in range(0,len(layerLengths)):
            self.weights.append((np.random.normal(size=(layerLengths[i], prev)) * np.sqrt(2/layerLengths[i])))
            prev = layerLengths[i]
            self.bias.append((np.random.normal(size=(prev, 1)) * np.sqrt(2/prev)))
        self.weights.append((np.random.normal(size=(10, prev)) * np.sqrt(2/10)))
        self.bias.append(np.random.normal(size=(10, 1)) * np.sqrt(2/10))
    
    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig
    def leaky_Relu(self,x):
        count = 0
        for val in x:
             x[count] = val*0.01 if val < 0 else val
             count+=1
        return x

    def  leaky_Relu_Derivative(self,x):
        count = 0
        for val in x:
            x[count] = 0.01 if val < 0 else 1
            count+=1
        return x
    def forwardProp(self):
        ##Start at input, multiply through weights, end at output. Similar structure to top
        self.hlVals.append(self.weights[0].dot(self.input) + self.bias[0])
        self.hlSig.append(self.sigmoid(self.hlVals[0]))
        for i in range(1,len(self.weights)):
            self.hlVals.append(self.weights[i].dot(self.hlSig[i-1]) + self.bias[i])
            self.hlSig.append(self.sigmoid(self.hlVals[i]))
            
        #self.hlVals.clear()
       #self.hlSig.clear()
    def sigdir(self,x):
        return self.sigmoid(x) * (1- self.sigmoid(x))

    def backProp(self, ind):
        
        prev = 2*(self.hlSig[len(self.hlSig)-1]-self.responseCleaner(self.answers[self.only123[ind]]))
        #prev = 2*(self.hlSig[len(self.hlSig)-1]-self.responseCleaner(self.answers[ind]))
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
        
        #After for loop, manually do for input layer, 0
        #DCA,DCZ,DCB,DCW
        #DCAn = sigderiv(Zn+1)
        #DCZn = WnAn-1 + bn
        #DCBn = ?
        #DCWn = An-1
            
    def responseCleaner(self, ans):
        
        retArr = []
        for x in range(0,10):
            if(x==ans):
                retArr.append(1)
            else :
                retArr.append(0)
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        #print(f"Guess: {guess}")
        #print(f"Ansr : {ans}")
        #self.options.append(np.argmax(self.hlSig[len(self.hlSig)-1]))
        if(guess==ans):
            self.correctList[ans]+=1
            self.correct +=1
        else :
            self.incorrectList[ans]+=1
        return np.reshape(retArr,(len(retArr),1))

    def update(self, alpha):
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
        

    def train(self,iterations, alpha, epoch, batch):
        data = []
       
        accuracy = []
        for x in range(0,epoch):
            print(f"Start Val: {self.only123[0:5]}")
            for i in range(0,iterations):
                
                self.trainInput(i)
                self.forwardProp()
                #print(self.hlSig[len(self.hlSig)-1])
                self.backProp(i)
                #Check to see if i%10 = 0, then set the sum equal to the DCoW and DCoB. else, sum += DCoW,
                if(i%batch==0):
                    self.sumCW = self.DCoW
                    self.sumCB = self.DCoB
                elif(i%batch==batch-1):
                    self.update(alpha)
                else:
                    self.summation(alpha)
                #At i%10==9,  
            
                self.clear()
                if(i%100==0):
                    
                    accuracy.append(self.correct/100)
                    print(self.correct/100)
                    self.correct = 0
            print(f"Next Epoch ----------------------")
        print(f"Correct Guesses: {self.correctList}")
        print(f"Incorrect Guesses: {self.incorrectList}")
        
    def updateInput(self, input):
        self.input = input
    def clear(self):
        self.hlVals.clear()
        self.hlSig.clear()
        self.DCoW.clear()
        self.DCoB.clear()
        #for i in range()
    def summation(self, alpha):
        
        for i in range(0,len(self.weights)):
            
            self.sumCW[i] += np.multiply(self.DCoW[i],alpha)
            
        
        for i in range(0,len(self.weights)):
            
            self.sumCB[i] += np.multiply(self.DCoB[i],alpha)
            
    def trainInput(self, ind):
        self.input =  np.reshape(self.data[self.only123[ind]][1:785],(784,1))
        #self.input =  np.reshape(self.data[ind][1:785],(784,1))
        for i in range(0,len(self.input)):
            val = self.input[i]
            if(val>=210):
                self.input[i]=1
            else :
                self.input[i]=0
   
    def returnWeights(self):
        return self.weights
        """
import numpy as np
from scipy.special import expit
import pandas as pd
#Potentially use leaky relu for weights and bias, and use sigmoid for output normalization.
#Avoiding vanishing gradient while keeping outputs scaled between -1 and 1
class NeuralNetwork:
    weights = []
    bias = []
    answers = []
    input = []
    hlVals = []
    hlSig = []
    output = []
    DCoW = []
    DCoB = []
    sumCW = []
    sumCB = []
    only123 = []
    options = []
    
    correct = 0
    correctList = [0,0,0,0,0,0,0,0,0,0]
    incorrectList = [0,0,0,0,0,0,0,0,0,0]
    
    data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
    data = np.array(data[1:50000])
    i = 0
    
    for x in data:
        answers.append(x[0])
        if(x[0]==0 or x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4):
            only123.append(i)
        i+=1
    
    outputLength = 10
    hiddenLayerLength = 0
    def __init__(self, layerLengths):
        prev = 784
        
        for i in range(0,len(layerLengths)):
            self.weights.append((np.random.normal(size=(layerLengths[i], prev)) * np.sqrt(2/layerLengths[i])))
            prev = layerLengths[i]
            self.bias.append((np.random.normal(size=(prev, 1)) * np.sqrt(2/prev)))
        self.weights.append((np.random.normal(size=(self.outputLength, prev)) * np.sqrt(2/self.outputLength)))
        self.bias.append(np.random.normal(size=(self.outputLength, 1)) * np.sqrt(2/self.outputLength))
    
    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig
    def leaky_Relu(self,x):
        count = 0
        for val in x:
             x[count] = val*0.01 if val < 0 else val
             count+=1
        return x

    def  leaky_Relu_Derivative(self,x):
        count = 0
        for val in x:
            x[count] = 0.01 if val < 0 else 1
            count+=1
        return x
    def forwardProp(self):
        length = len(self.weights)
        ##Start at input, multiply through weights, end at output. Similar structure to top
        self.hlVals.append(self.weights[0].dot(self.input) + self.bias[0])
        self.hlSig.append(self.leaky_Relu(self.hlVals[0]))
        for i in range(1,length-1):
            self.hlVals.append(self.weights[i].dot(self.hlSig[i-1]) + self.bias[i])
            self.hlSig.append(self.leaky_Relu(self.hlVals[i]))
        self.hlVals.append(self.weights[length-1].dot(self.hlSig[length-2]) + self.bias[length-1])
        self.hlSig.append(self.sigmoid(self.hlVals[length-1]))
        #self.hlVals.clear()
       #self.hlSig.clear()
    def sigdir(self,x):
        return self.sigmoid(x) * (1- self.sigmoid(x))

    def backProp(self, ind):
        length = len(self.hlSig)-1
        prev = 2*(self.hlSig[length]-self.responseCleaner(self.answers[self.only123[ind]]))
        #prev = 2*(self.hlSig[length]-self.responseCleaner(self.answers[ind]))
        DAoZn = self.leaky_Relu_Derivative(prev)   
        DZoWn = np.reshape(self.hlSig[length-1],(1,len(self.hlSig[length-1])))
        DZnA1 = np.reshape(self.weights[length],(len(self.hlSig[length-1]),len(self.hlSig[length])))
        dot = DAoZn*prev
        self.DCoW.append(dot*DZoWn)
        self.DCoB.append(dot)
        prev = DZnA1.dot(dot)
        for i in range(len(self.hlSig)-2,0,-1):
            
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
        
        #After for loop, manually do for input layer, 0
        #DCA,DCZ,DCB,DCW
        #DCAn = sigderiv(Zn+1)
        #DCZn = WnAn-1 + bn
        #DCBn = ?
        #DCWn = An-1
            
    def responseCleaner(self, ans):
        
        retArr = []
        for x in range(0,10):
            if(x==ans):
                retArr.append(1)
            else :
                retArr.append(0)
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        #print(f"Guess: {guess}")
        #print(f"Ansr : {ans}")
        #self.options.append(np.argmax(self.hlSig[len(self.hlSig)-1]))
        if(guess==ans):
            self.correctList[ans]+=1
            self.correct +=1
        else :
            self.incorrectList[ans]+=1
        return np.reshape(retArr,(len(retArr),1))

    def update(self, alpha):
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
        

    def train(self,iterations, alpha, epoch, batch):
        data = []
       
        accuracy = []
        for x in range(0,epoch):
            print(f"Start Val: {self.only123[0:5]}")
            for i in range(0,iterations):
                
                self.trainInput(i)
                self.forwardProp()
                
                self.backProp(i)
                #Check to see if i%10 = 0, then set the sum equal to the DCoW and DCoB. else, sum += DCoW,
                if(i%batch==0):
                    self.sumCW = self.DCoW
                    self.sumCB = self.DCoB
                elif(i%batch==batch-1):
                    self.update(alpha)
                else:
                    self.summation(alpha)
                #At i%10==9,  
            
                
                if(i%100==0):
                    #print(self.hlSig[len(self.hlSig)-1])
                    accuracy.append(self.correct/100)
                    print(self.correct/100)
                    self.correct = 0
                self.clear()
            print(f"Next Epoch ----------------------")
        print(f"Correct Guesses: {self.correctList}")
        print(f"Incorrect Guesses: {self.incorrectList}")
        
    def updateInput(self, input):
        self.input = input
    def clear(self):
        self.hlVals.clear()
        self.hlSig.clear()
        self.DCoW.clear()
        self.DCoB.clear()
        #for i in range()
    def summation(self, alpha):
        
        for i in range(0,len(self.weights)):
            
            self.sumCW[i] += np.multiply(self.DCoW[i],alpha)
            
        
        for i in range(0,len(self.weights)):
            
            self.sumCB[i] += np.multiply(self.DCoB[i],alpha)
            
    def trainInput(self, ind):
        self.input =  np.reshape(self.data[self.only123[ind]][1:785],(784,1))
        #self.input =  np.reshape(self.data[ind][1:785],(784,1))
        for i in range(0,len(self.input)):
            val = self.input[i]
            if(val>=210):
                self.input[i]=1
            else :
                self.input[i]=0
   
    def returnWeights(self):
        return self.weights
"""