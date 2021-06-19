# install opencv by typing pip install opencv-python in the python terminal 
#import opencv library(cv2)

import cv2 
# Using Haar cascade algorithm to detect the face download it from opencv github account

face_cascade=cv2.CascadeClassifier('External_Libraries\haarcascade_frontalface_default.xml')

capture=cv2.VideoCapture(0) #For opening the system camera for capturing the video we use 0 for other type give the path
while capture.isOpened(): 
    
    ret,img=capture.read() #To read the captured video  
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Making the image frame to gray scale for better results
    faces=face_cascade.detectMultiScale(gray,1.1,10)   #For detecting the multiple faces  

    for(x,y,w,h) in faces:  #Looping through captured faces by taking there dimensions
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3) #Drawing rectangular shape on the detected image

    cv2.imshow('img',img)  #Displaying the recatngular shape on the detected face
    
    if cv2.waitKey(1) & 0xFF==ord('q'): # Close the camera window if we press the q
        break
capture.release() #Closes video file or capturing device. The method is automatically called by subsequent VideoCapture::open and by VideoCapture destructor. The C function also deallocates memory and clears *capture pointer.





    

