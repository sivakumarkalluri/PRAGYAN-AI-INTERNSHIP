import face_recognition
import numpy as np
import cv2
import os
path = 'Images'
images = []
known_face_names = []
myList = os.listdir(path)
print(myList)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    known_face_names.append(os.path.splitext(cl)[0])
print(known_face_names)
known_face_encodings = []

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encode)
video_recording = cv2.VideoCapture(0)

while True:
    success, img = video_recording.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS,model="cnn")
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame,num_jitters=100)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(known_face_encodings, encodeFace,tolerance=0.50)
        faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)
        
        name="Unknown"
        if True in matches:
            matchIndex = np.argmin(faceDis)
            name=known_face_names[matchIndex]

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 6)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_recording.release()
cv2.destroyAllWindows()


