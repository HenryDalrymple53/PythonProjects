import numpy
import cv2

capture=cv2.VideoCapture(0)

cascade_classifier = cv2.CascadeClassifier("C:\\Users\\henry\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")

while True:
    
    
    ret,frame=capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray,100,200)
    gaussian = cv2.GaussianBlur(canny, (5, 5), 10)
    
    contours,hierarchy =  cv2.findContours(gaussian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    largest = 0
    ind = 0
    if(len(contours) > 0):
        for x in contours:
            if(cv2.contourArea(contours[ind]) > cv2.contourArea(contours[largest])):
                largest = ind
            ind+=1
           
            
        cv2.drawContours(frame, contours[largest], -1, (255,0,0),5)
        #print("Break between prints")
        #print(contours[largest][0][0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.imshow('Color',frame)
capture.release()
cv2.destroyAllWindows()

#faces = cascade_classifier.detectMultiScale(image_grey, minSize=(30, 30))
 #for (x, y, w, h) in faces:q
    #    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)