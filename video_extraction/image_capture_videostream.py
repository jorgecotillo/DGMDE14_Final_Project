import cv2
from scipy.spatial import distance
from mlxtend.image import extract_face_landmarks
import os
import numpy as np
from random import sample

DESIRED_HEIGHT = 128
DESIRED_WIDTH = 128
capture = False
counter = 0
VIDEO_NUMBER_OF_INTEREST = [0]
MOUTH_OR_EYE = 'eye'
def resize(image):
  h, w = image.shape[:2]
  img = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT))
  return img

def get_frame(sec):
  start = 0
  vid.set(cv2.CAP_PROP_POS_MSEC,start+sec*70)
  frames,image = vid.read()
  return frames,image

def is_eye_closed(data):
  dis1 = distance.euclidean(data[1],data[5])
  dis2 = distance.euclidean(data[2],data[4])
  dis3 = distance.euclidean(data[7],data[11])
  dis4 = distance.euclidean(data[8],data[10])
  dis = np.average([dis1,dis2,dis3,dis4])
  print(dis)
  return True if dis <= 25 else False

def is_mouth_closed(data):
  dis1 = distance.euclidean(data[2],data[10])
  dis2 = distance.euclidean(data[3],data[9])
  dis3 = distance.euclidean(data[4],data[8])
  dis = np.average([dis1,dis2,dis3])
  # print(dis)
  return True if dis <= 100 else False

# For webcam input:
root_path = os.path.abspath('../videos')
extracted_images_path = os.join(root_path, 'extracted_images')
num = 0
current_number_of_pictures = 0
max_number_of_pictures = 2000
videos_folder = os.listdir(root_path)

while current_number_of_pictures < max_number_of_pictures:
  for video_folder in videos_folder:
    if current_number_of_pictures >= max_number_of_pictures:
      break
    
    video_path = os.path.join(root_path, video_folder)

    # sample_video_path is a folder containing all videos to capture
    # using this folder structure to comply with another big dataset
    # we have
    for sample_video_folder_path in os.listdir(video_path):
      if current_number_of_pictures >= max_number_of_pictures:
        break
      
      sample_video_path = os.path.join(video_path, sample_video_folder_path)
      sample_video_file_names = os.listdir(sample_video_path)

      for sample_video_file_name in sample_video_file_names:
        if current_number_of_pictures >= max_number_of_pictures:
          break
        
        sample_video_file_path = os.path.join(sample_video_path, sample_video_file_name)
        wantplot = False
        vid = cv2.VideoCapture(sample_video_file_path)
        images_captured_count = 0
        max_images_to_capture = 1000
        sec = 0
        success,img = get_frame(sec)

        while success and images_captured_count <= max_images_to_capture and current_number_of_pictures <= max_number_of_pictures:
          landmark = extract_face_landmarks(img)
          try:
            if MOUTH_OR_EYE=='eye': 
              is_drowsy = is_eye_closed(landmark[36:48,1])
            elif MOUTH_OR_EYE=='mouth':
              is_drowsy = is_mouth_closed(landmark[48:88,1])

          except TypeError:
            if wantplot:
              import matplotlib.pyplot as plt
              plt.imshow(img)
              print("Error at count = {} at time equals {} seconds".format(images_captured_count,sec))
              plt.show()
              wantplot = False
              is_drowsy = False
            else:
              print("Error at count = {} at time equals {} seconds".format(images_captured_count,sec))
              is_drowsy = False

          if is_drowsy:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            import datetime
            now = datetime.now()
            file_name = "{0}_{1}.jpg".format(now.strftime("%m_%d_%Y_%H%M%S%f"), num)
            cv2.imwrite(os.path.join(extracted_images_path, file_name),gray_img)
            num += 1
            current_number_of_pictures += 1
          images_captured_count += 1
          sec += 1
          sec = round(sec,2)

          if sec%100==0:
            print('Video steaming at --> {} seconds\n Folder = {}\n folder number = {}'.format(sec, sample_video_path , sample_video_file_path))
          success,img = get_frame(sec)

  vid.release()