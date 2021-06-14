from scipy.spatial import distance
from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd

class DrowsinessDetection(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.eye = None
        self.path = None
        #self.calibration = Calibration()

        # _face_detector is used to detect faces
        #self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        #cwd = os.path.abspath(os.path.dirname(__file__))
        #model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        #self._predictor = dlib.shape_predictor(model_path)
    def sound_alarm(self, path):
                #play an alarm sound                  
                playsound.playsound(str(path))
	
    def eye_aspect_ratio(self, eye):
        
                A = distance.euclidean(eye[1], eye[5])
                B = distance.euclidean(eye[2], eye[4])
                C = distance.euclidean(eye[0], eye[3])
                ear = (A + B) / (2.0 * C)
                return ear
    def amplitude(self, eye):
        
                A = distance.euclidean(eye[1], eye[5])
                B = distance.euclidean(eye[2], eye[4])
                amp = (A + B) / 2.0
                return amp

