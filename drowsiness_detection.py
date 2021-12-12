from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np

from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

import smtplib

import os
import joblib

from scipy.spatial import distance

import math

wake_up_sound_path = os.path.abspath('wake_up.mp3')

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Alert", "Tired"]

    def __init__(self, model_json_file, model_weights_file):        
        self.knn_model = joblib.load('knn.pkl')
        # load model from JSON file
        # with open(model_json_file, "r") as json_file:
        #    loaded_model_json = json_file.read()
        #    self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        # self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model.make_predict_function()

    def predict_emotion(self, features):
        self.preds = self.knn_model.predict(features)
        print(self.preds)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

class EmailNotification(object):

    def send_email(self, to_email):
        # TODO: Use environment variables
        gmail_user = 'drowsiness.detection.no.reply@gmail.com'
        gmail_password = 'raspberrypi2021'

        sent_from = gmail_user
        to = [to_email]
        subject = 'Please wake up!'
        body = "Hey, you need to wake up please!!"

        email_text = """\
        From: %s
        To: %s
        Subject: %s

        %s
        """ % (sent_from, ", ".join(to), subject, body)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(gmail_user, gmail_password)
            server.sendmail(sent_from, to, email_text)
            server.close()

            print("Email sent!")

        except Exception as ex:
            print(ex)
            print("Something went wrong...")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def main():
    camera =  PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture =  PiRGBArray(camera, size=(640, 480))

    # Add some delay before - we had issues with a race condition where the camera is not ready yet.
    time.sleep(0.1)
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = FacialExpressionModel("model.json", "model_weights.h5")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Open the camera and start streaming frames
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        # Capturing image from camera frame
        image = frame.array
        
        # Converting image into gray scale (to make the detection easier)
        # TODO: What easier means?
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect a face
        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Capture the face
            face = gray_frame[y:y+h, x:x+w]

            # Resize image - our training model was done in 48x48 pixels
            resized_image = cv2.resize(face, (256, 256))

            # Extract landmarks
            landmarks = extract_face_landmarks(resized_image)
            data = [landmarks]
            data = np.array(data)

            features = []
            for d in data:
                eye = d[36:68]
                ear = eye_aspect_ratio(eye)
                mar = mouth_aspect_ratio(eye)
                cir = circularity(eye)
                mouth_eye = mouth_over_eye(eye)
                features.append([ear, mar, cir, mouth_eye])

            # Predict the expression
            expression_prediction = model.predict_emotion(features)

            # Notification section
            if expression_prediction.lower() == 'neutral':

                # Email notification
                email_notification = EmailNotification()
                email_notification.send_email('jorge.cotillo@gmail.com')

                # Play sound notification
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(wake_up_sound_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() == True:
                    continue
            
            cv2.putText(image, expression_prediction, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('Expression recognized', image)

        key = cv2.waitKey(1)  & 0xFF
        rawCapture.truncate(0)

        if key == ord('q'):
            exit(1)

if __name__ == '__main__':
    main()