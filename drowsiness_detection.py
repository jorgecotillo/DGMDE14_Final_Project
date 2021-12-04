from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import io
import cv2
import numpy as np
from .facial_expression_model import FacialExpressionModel

def main():
    camera =  PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture =  PiRGBArray(camera, size=(640, 480))

    # Add some delay before - we had issues with a race condition where the camera is not ready yet.
    time.sleep(0.1)
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = FacialExpressionModel("./model.json", "./model_weights.h5")
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
            resized_image = cv2.resize(face, (48, 48))

            # Predict the expression
            expression_prediction = model.predict_emotion(resized_image[np.newaxis, :, :, np.newaxis])
            
            cv2.putText(image, expression_prediction, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('Expression recognized', image)

        key = cv2.waitKey(1)  & 0xFF
        rawCapture.truncate(0)

        if key == ord('q'):
            exit(1)

if __name__ == '__main__':
    main()