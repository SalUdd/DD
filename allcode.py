
import cv2
import numpy as np
from deepface import DeepFace
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from pygame import mixer

def calculate_EYE(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio

#variables for current states
sleeping = 0
drowsy = 0
active = 0
status = " "
color = (0,0,0)

#audio functions
mixer.init()
sound = mixer.Sound('moan.wav')

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
#classifier =load_model('./Emotion_Detection.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 1)

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

    
    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_eye = calculate_EYE(leftEye)
        right_eye = calculate_EYE(rightEye)

        EYE = (left_eye+right_eye)/2
        EYE = round(EYE,2)
        #telling the system what to do incase of an eye blink detected
        if(EYE<0.3):
            sleeping+=1
            drowsy=0
            active=0
            if(sleeping>6):
                status="sleepin!!!!!"
                if sound.get_num_channels() == 0:
                    sound.play()
                color = (255,0,0)
        
        else:
            drowsy = 0
            sleeping = 0
            active += 1
            if(active>6):
                status="Active!"
                sound.stop()
                color = (0,0,255)
                
        #displaying the output on the screen        
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
        
      
            
            

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











