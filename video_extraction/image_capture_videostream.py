import cv2; from scipy.spatial import distance
from mlxtend.image import extract_face_landmarks
import os; import numpy as np
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

if not os.path.exists("C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\drowsy"):
  os.mkdir("C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\drowsy",777)
# For webcam input:
root_path = "C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\raw_unzip\\"
num=0
final_num = 0
# list_folder = sample(os.listdir(root_path),len(os.listdir(root_path)))
list_folder = os.listdir(root_path)
while final_num<621:
  for folder in list_folder:
    if final_num>=621:
      break
    # list_foldNumber = sample(os.listdir(root_path+'\\'+folder),len(os.listdir(root_path+'\\'+folder)))
    list_foldNumber = sample(os.listdir(root_path+'\\'+folder),len(os.listdir(root_path+'\\'+folder)))
    for foldNumber in list_foldNumber:
      if final_num>=621:
        break
      # if foldNumber=='03':
      #   print('Skipping {} folder at {} folder number'.format(folder,foldNumber))
      #   continue
      # for i in VIDEO_NUMBER_OF_INTEREST:
      last_path = os.path.join(root_path,folder,foldNumber)
      list_filename = os.listdir(last_path)
      for name in list_filename:
        if final_num>=2000:
          break
        # path= os.path.join(root_path,folder,foldNumber,str(i)+'.MOV')
        path= os.path.join(root_path,folder,foldNumber,name)
        wantplot=False
        vid = cv2.VideoCapture(path)
        count=0
        sec = 0
        success,img = get_frame(sec)
        while success and count <= 500 and final_num<=2000:
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
              print("Error at count = {} at time equals {} seconds".format(count,sec))
              plt.show()
              wantplot = False
              is_drowsy = False
            else:
              print("Error at count = {} at time equals {} seconds".format(count,sec))
              is_drowsy = False
          if is_drowsy:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join('C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\drowsy\\',str(num)+'.jpg'),gray_img)
            num+=1
            final_num+=1
          count +=1
          sec = sec + 1
          sec = round(sec,2)
          if sec%100==0:
            print('Video steaming at --> {} seconds\n Folder = {}\n folder number = {}'.format(sec,folder,foldNumber))
          success,img = get_frame(sec)

  vid.release()