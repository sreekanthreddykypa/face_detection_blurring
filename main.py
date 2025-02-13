import cv2
import mediapipe as mp


# read image
img_path = 'C:\Face_detection_blurring\data\testImg.png'

img = cv2.imread(img_path)
if img is None:
    print("Error: Image not found or could not be loaded.")

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence = 0.5) as face_detection:
  img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  out = face_detection.process(img_rgb)

  print(out.detections)
