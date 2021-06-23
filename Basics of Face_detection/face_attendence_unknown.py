import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime
paths=["Images/Known_images","Images/Unknown_images"]
from PIL import Image 
import PIL 
images=[]
photonames=[]
Unknown_names=[]
for index,path in enumerate(paths):
    myList=os.listdir(paths[index])
    for items in myList:
        print(items)
        curImg=cv2.imread(f'{path}/{items}')
        
        images.append(curImg)
        photonames.append(os.path.splitext(items)[0])
        if path=="Images/Unknown_images":
            Unknown_names.append(os.path.splitext(items)[0])
if len(Unknown_names)>0:
    Unknown_name_last=os.path.splitext(Unknown_names[-1])[0]
    print(Unknown_name_last)


    Unknown_last=int(Unknown_name_last[7:])
    print(Unknown_last)
elif len(Unknown_names)==0:
    Unknown_last=0
print(photonames)

known_face_encodings=[]
def findEncodings(images):
    
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        try: 
            encode=face_recognition.face_encodings(img)[0]
            if len(encode)>0:
                known_face_encodings.append(encode)

        
                
        except IndexError as e:
                continue
        
        
    
findEncodings(images)
def MarkAttendance(name):
    with open ('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            Time_now=datetime.now()
            time_string=Time_now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_string}')

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
             
             
        
        y1,x2,y2,x1=facelocations
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2+30,y2+30),(0,255,0),2)
       

        if name not in photonames: 
            try:
                img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                encode1=face_recognition.face_encodings(img1)[0]
                if len(encode1)>0:
                    Unknown_last+=1
                    name="Unknown"+str(Unknown_last)
               
                    cv2.imwrite(f'{paths[1]}/{name}.jpg', img)
                    known_face_encodings.append(encode1)
                    print(f'{name} face detected ')
                    
            
                    photonames.append(name)
                
            except IndexError as e:
                
                print("Face is not detecting wait sometime.......")
                    
                continue

            
            
            
            
            # Unknown_list.append(name)
           
        


        cv2.rectangle(img,(x1,y2-5),(x2+30,y2+30),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        print(f'{name} face detected ')
        MarkAttendance(name)
    cv2.imshow('face_rcognisation',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()