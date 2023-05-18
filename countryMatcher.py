import pandas as pd
import numpy as np
class Number:
    startPos = 0
    curPos = 0
    curValue = 0
    def __init__(self,startPos,curPos,value):
        self.startPos = startPos
        self.curPos = curPos
        self.value = value
data = pd.read_csv("C:\\PythonProjects\\DataFiles\\input.txt")
#data = np.array(data)
data = [1,2,-3,3,-2,0,4]
arr = []

ind = 0
for x in data:
    arr.append(Number(ind,ind,(int(x))))
    
    ind+=1
newArr = arr[:]
n=0
for x in arr:
    print(x.startPos)

