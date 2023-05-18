import numpy as np
import csv
import sys
import pandas as pd
import random
from student import *
sys.path.append("C/PythonProjects/countryMatcher")
data = pd.read_csv("C:/PythonProjects/countryMatcher/countries.csv")
names = pd.read_csv("C:/PythonProjects/countryMatcher/names.csv")
studentNames = np.array(names)
data = np.array(data)
slotsTaken = []
numOptions = 5
countryStudents = []
countryRange = 50

for x in data:
    slotsTaken.append(0)
    countryStudents.append([-1,-1,-1])
    
countries = data
#len(countries)-1
totalChoices = []
numChoices = []
students = []
count = 0
studentIndex = []
for x in studentNames:
    studentIndex.append(count)
    studentChoices = []
    
    for i in range(0,numOptions):
        rand = random.randint(0,countryRange)
        while(rand in studentChoices):
            rand = random.randint(0,countryRange)
        studentChoices.append(rand)
    s = student(x,studentChoices)
    students.append(s)
    
    numChoices.append(0)
    count+=1
def numOfN1(arr):
    
    count = 0
    for x in arr:
        if x == -1:
            count+=1
    return 3-count

runningIndex = studentIndex.copy()

for i in range(0,numOptions):
    studentIndex = runningIndex.copy()
    runningIndex = studentIndex.copy()
    delNum = 0
    for x in studentIndex:
        s = students[x]
        if(slotsTaken[s.prefs[i]]!=3):
            del runningIndex[delNum]
            country = countryStudents[s.prefs[i]]
            country[numOfN1(country)] = s
            s.happiness = 5-i
            slotsTaken[s.prefs[i]]+=1
        else:
            delNum+=1
    

print(sum(slotsTaken))

#def findNswap(self,cNum):
    #for x in countryStudents[cNum]:


happiness = 0
c = 0


#To further condense countries into coherent groups of three, trickle down people from single person groups first, finding another 
#group under 3 big, and adding themselves to it.


#To optimize, if student got pref0, loop through pref1,pref2,pref3. if pref1, loop pref2,pref3, etc
#loop through all countries in above prefs, and total happiness increases in options, switch people.
#Recursion loop to maximize happiness
for x in range(0,countryRange):
    
    if(slotsTaken[x]>0):
        
        for s in countryStudents[x]:
            if(s != -1):
                c+=1
                happiness += s.happiness
                print(f"{countries[x][1]} {s.name} {s.happiness}" )
            
            #for p in countryStudents[s.prefs[1]]:
print(happiness/c)
print(c)
