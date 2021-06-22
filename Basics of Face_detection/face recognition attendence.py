import cv2
import numpy as np
import face_recognition 
import os
path='Images'
images=[]
photonames=[]
myList=os.listdir(path)
print(myList)
for items in myList:
    curImg=cv2.imread(f'{path}/{items}')
    images.append(curImg)
    photonames.append(os.path.splitext(items)[0])
def findEncodings(images):
    encodings=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings
known_face_encodings=findEncodings(images)
video_capture=cv2.VideoCapture(0)
while True:
    ret,img=video_capture.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faces_loc_frame=face_recognition.face_locations(imgS,model="cnn")
    encodings_frame=face_recognition.face_encodings(imgS,faces_loc_frame)
    for encodesFace,facelocations in zip(encodings_frame,faces_loc_frame):
        matches=face_recognition.compare_faces(known_face_encodings,encodesFace,tolerance=0.50)
        # print(matches)
        faceDistance=face_recognition.face_distance(known_face_encodings,encodesFace)
       
        # print(matchIndex)
        name="Unknown"
        if True in matches:
             matchIndex=np.argmin(faceDistance)
             
             name=photonames[matchIndex]
             print(name)
        y1,x2,y2,x1=facelocations
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2+30,y2+30),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-5),(x2+30,y2+30),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('face_rcognisation',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
