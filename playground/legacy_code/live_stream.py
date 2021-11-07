from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import io
import cv2
import numpy as np
import logging
import socketserver
import pickle
from threading import Condition
from http import server
from datetime import date, datetime

PAGE="""\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.image = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):

            a = buf.find(b'\xff\xd8')
            b = buf.find(b'\xff\xd9')

            if a != -1 and b != -1:
                _jpg = buf[a:b+2]
                _bytes = buf[b+2:]
                self.image = cv2.imdecode(np.fromstring(_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                        gray = cv2.cvtColor(output.image, cv2.COLOR_BGR2GRAY)
                        faces = faceCascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30))

                        for (x, y, w, h) in faces:
                            now = datetime.now()
                            print('face recognized' + now.strftime("%m/%d/%Y, %H:%M:%S"))
                            cv2.rectangle(output.image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

output = StreamingOutput()

def main():
    camera =  PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture =  PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)

    camera.start_recording(output, format='mjpeg')
    
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        print(type(frame))
        print(type(frame.array))
        image = frame.array
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        for (x, y, w, h) in faces:
            print('face recognized')
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('frame', image)

        key = cv2.waitKey(1)  & 0xFF
        rawCapture.truncate(0)

        if key == ord('q'):
            exit(1)

if __name__ == '__main__':
    main()
