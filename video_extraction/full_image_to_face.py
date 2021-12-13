import cv2
import mediapipe as mp; import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

DESIRED_HEIGHT = 48
DESIRED_WIDTH = 48
capture = False
counter = 0
def resize(image):
  h, w = image.shape[:2]
  img = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT))
  return img
num=0
# For webcam input:
root_path = "C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\alert_full_size"
print(len(os.listdir(root_path)))
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  for path in os.listdir(root_path):
    image = cv2.imread(os.path.join(root_path, path))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.detections:
      for detection in results.detections:
        shape = image.shape 

        relative_x = int(detection.location_data.relative_bounding_box.xmin* shape[1])
        relative_y = int(detection.location_data.relative_bounding_box.ymin * shape[0])
        relative_w = int(detection.location_data.relative_bounding_box.width* shape[1])
        relative_h = int(detection.location_data.relative_bounding_box.height * shape[0])
        # mp_drawing.draw_detection(image, detection)
        # cv2.circle(image,center = (relative_x,relative_y),radius = 20,color = (255,0,0))
        if relative_x > DESIRED_HEIGHT and relative_y > DESIRED_WIDTH and (relative_y+relative_w) < shape[1] and (relative_x+relative_h) < shape[0]:
            img = image[relative_y-DESIRED_WIDTH:relative_y+relative_w+DESIRED_WIDTH,relative_x-DESIRED_HEIGHT:relative_x+relative_h+DESIRED_HEIGHT]
            resize_img = resize(img)
            cv2.imwrite(("C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\alert_face\\"+str(num)+".jpg"),resize_img)
            num+=1
        else:
            resize_img = resize(image)
            cv2.imwrite(("C:\\Users\\nanda\\Documents\\harvard\\DGMDE14\\Final_Project\\sample_data\\texas_uofta_dataset\\process_data\\alert_face\\"+str(num)+".jpg"),resize_img)
            num+=1     
    # Flip the image horizontally for a selfie-view display.
    # if cv2.waitKey(5) & 0xFF == 32:
    #     print('entered')
    #     counter += 1
    #     filename = 'test_'+ str(counter) + '.jpg'
    #     cv2.imwrite(filename,resize_img)
    #     continue
    # #cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break
# cap.release()