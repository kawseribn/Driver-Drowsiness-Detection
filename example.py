"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from Drowsiness_Detection import DrowsinessDetection
from scipy.spatial import distance
from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
import numpy as np
from playsound import playsound
import argparse
import imutils
import time
import dlib
import pandas as pd

gaze = GazeTracking()
DD = DrowsinessDetection()
webcam = cv2.VideoCapture(0)
thresh = 0.25
frame_check = 24

COUNTER = 0
ALARM_ON = False

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(1)
Status = -1
flag=0
time.sleep(1.0)
df = []
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""
    #ret, frame1=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        
                        shape = predict(gray, subject)
                        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = DD.eye_aspect_ratio(leftEye)
                        rightEAR = DD.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        ampli1 = DD.amplitude(leftEye)
                        ampli2 = DD.amplitude(rightEye)
                        total_ampli = (ampli1 + ampli2) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                        if gaze.is_blinking():
                            text = "Looking Down"
                            if gaze.lower_right():
                                text = "Looking Lower Right"
                            elif gaze.lower_left():
                                text = "Looking Lower Left"
                        elif gaze.is_right():
                                text = "Looking right"
                        elif gaze.is_left():
                                text = "Looking left"
                        elif gaze.is_center():
                                text = "Looking center"
                        elif gaze.is_up():
                                text = "Looking UP"
                        if (text=="Looking right"):
                                Status = 0
                        if (text=="Looking left"):
                                Status = 1
                        if (text=="Looking UP"):
                                Status = 2
                        if (text=="Looking Down"):
                                Status = 3
                        if (text=="Looking Lower Right"):
                                Status = 30
                        if (text=="Looking Lower Left"):
                                Status = 31
                        if ear < thresh:
                                flag += 1
                                if (1==1):
                                        df.append(
                                                {
                                                        'Amplitude': total_ampli,
                                                        'Left_Ear': leftEAR,
                                                        'Right_Ear': rightEAR,
                                                        'Ear': ear,
                                                        'Eye_closed': flag,
                                                        'Gaze_Status': Status
                        }
                    )
                                print (str(flag) +" observing "+ str(ear)+" Status = "+str(Status))
                                if flag >= frame_check:
                                        # if the alarm is not on, turn it on
                                        # check to see if an alarm file was supplied,
                                        # and if so, start a thread to have the alarm
                                        # sound played in the background
                                        DD.sound_alarm("alarm.mp3")
                                
                                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        cv2.putText(frame, "****************ALERT!****************", (10,325),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        #print ("Drowsy")
                                        #pd.DataFrame(df)
                        #df = pd.DataFrame(df)
                        #df.to_csv (r'D:\AAAAAAAAAA\fileone.csv', index = False, header=True)
                        else:
                                flag = 0
                                ALARM_ON = False
                        cv2.imshow("Frame", frame)
    

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    #if cv2.waitKey(1) == 27:
        #break
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        df = pd.DataFrame(df)
        df.to_csv (r'D:\AAAAAAAAAA\kawser1.csv', index = False, header=True)#'D:\AAAAAAAAAA\fileone1.csv' er jaygay tor laptop er ekta location dibi,jeikhane dataset save hobe
        break
cv2.destroyAllWindows()
