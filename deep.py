
import cv2
import numpy as np
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
#classifier =load_model('./Emotion_Detection.h5')

cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(
        frame,
        result[0]['dominant_emotion'],
        (50,50),
        font, 3,
        (0,0,255),
        2,
        cv2.LINE_4)

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











