import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from pygame import mixer

from keras.models import load_model
from time import sleep
from keras.utils.image_utils import img_to_array

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('network-5Labels.h5')

class_labels = ['Surprise', 'Neutral', 'Negative', 'Happy', 'Sad']


def calculate_EYE(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio


# variables for current states
sleeping = 0
active = 0
status = " "
color = (0, 0, 0)
NegEmo = 0
NrmEmo = 0

# audio functions
mixer.init()
mixer.set_num_channels(4)
sound = mixer.Sound('alert.wav')
sound2 = mixer.Sound('sleepiness_alert.wav')
sound3 = mixer.Sound('emotion_alert.wav')

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 2)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_eye = calculate_EYE(leftEye)
        right_eye = calculate_EYE(rightEye)

        EYE = (left_eye+right_eye)/2
        EYE = round(EYE, 2)

        # telling the system what to do incase of an eye blink detected
        if EYE < 0.16:
            if label:
                if preds.argmax() == 4:
                    label = class_labels[1]
                else:
                    label = class_labels[preds.argmax()]
            sleeping += 1
            active = 0
            if sleeping > 24:  # 12 seconds
                mixer.Channel(0).stop()
                if sound2.get_num_channels() == 0:
                    mixer.Channel(1).set_volume(1)
                    mixer.Channel(1).play(sound2)
            elif sleeping > 14:  # 7 seconds
                mixer.Channel(0).set_volume(0.5)
            elif sleeping > 8:  # 4 seconds
                if sound.get_num_channels() == 0:

                    mixer.Channel(0).set_volume(0.2)
                    mixer.Channel(0).play(sound)
                color = (255, 0, 0)
        else:
            sleeping = 0
            active += 1
            if active > 2:
                status = "Active!"
                mixer.Channel(0).stop()
                mixer.Channel(1).stop()
                color = (0, 0, 255)

        # displaying the output on the screen
        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    faces2 = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (245, 135, 66), 2)
        roi_gray = gray[y+5:y+h-5, x+20:x+w-20]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            print("\nprediction = ", preds)

            if preds.argmax() == 4:
                label = class_labels[1]
            else:
                label = class_labels[preds.argmax()]

            # label = class_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            label_position = (x, y)
            # if preds.argmax() in [0, 3]:
            if preds.argmax() in [2]:
                NegEmo += 1
                if NegEmo > 2:
                    if sound3.get_num_channels() == 0:
                        mixer.Channel(2).play(sound3)
            elif preds.argmax() in [0, 1, 3]:
                NrmEmo += 1
                if NrmEmo > 2:
                    mixer.Channel(2).stop()

            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")

        cv2.imshow("Frame", frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
