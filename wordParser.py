import pandas as pd

read = open("C:\\PythonProjects\\DataFiles\\words.txt")
write = open("C:\\PythonProjects\\DataFiles\\output.txt", 'a')
lines = read.readlines()

for x in lines:
    x = x[0:5]
    s = ""
    s+='\"'
    s+=x
    s+="\","
    write.write(s)
    print(s)
