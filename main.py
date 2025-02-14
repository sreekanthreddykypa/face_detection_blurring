import os
import cv2
import mediapipe as mp
import argparse




# read image

def process_image(img,face_detection):
  H, W, _ = img.shape

  img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  out = face_detection.process(img_rgb)
  

  

  if out.detections is not None:

    for detection in out.detections:
      location_data = detection.location_data
      bbox = location_data.relative_bounding_box

      x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

      x1 = int(x1*W)
      y1 = int(y1*H)
      w = int(w*W)
      h = int(h*H)

      #blur

      img[y1:y1+h,x1:x1+w,:] = cv2.blur(img[y1:y1+h,x1:x1+w,:],(50,50))
  return img

      #img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),10)
  #cv2.imshow('img',img)
  #cv2.waitKey(0)

parser = argparse.ArgumentParser(description= "Face Detection and Blurring using MediaPipe")
parser.add_argument("--mode", type=str, required=True, choices=["image", "video", "webcam"],
                    help="Select mode: 'image', 'video', or 'webcam'")
parser.add_argument("--filepath", type=str, default=None,
                    help="File path for image or video (not required for webcam)")
args = parser.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
   os.makedirs(output_dir)

# detect faces
mp_face_detection = mp.solutions.face_detection

# Process Image Mode
if args.mode == "image":
    if not args.filepath:
        print("Error: --filepath is required for 'image' mode.")
        exit()
    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' does not exist.")
        exit()

    img = cv2.imread(args.filepath)
    if img is None:
        print("Error: Could not load image. Check file format.")
        exit()

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        img = process_image(img, face_detection)

    output_path = os.path.join(output_dir, "output.png")
    cv2.imwrite(output_path, img)
    print(f"Processed image saved at {output_path}")

# Process Video Mode
elif args.mode == "video":
    if not args.filepath:
        print("Error: --filepath is required for 'video' mode.")
        exit()
    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' does not exist.")
        exit()

    cap = cv2.VideoCapture(args.filepath)
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read video file. Check format.")
        exit()

    output_video = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   25, (frame.shape[1], frame.shape[0]))

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while ret:
            frame = process_image(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

    cap.release()
    output_video.release()
    print("Processed video saved in 'output/output.mp4'.")

# Process Webcam Mode
elif args.mode == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be accessed.")
        exit()

    ret, frame = cap.read()
    output_video = cv2.VideoWriter(os.path.join(output_dir, "output_webcam.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   25, (frame.shape[1], frame.shape[0]))

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while ret:
            frame = process_image(frame, face_detection)
            cv2.imshow("Webcam Face Blur", frame)

            if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
                break

            output_video.write(frame)
            ret, frame = cap.read()

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    print("Webcam session ended. Processed video saved in 'output/output_webcam.mp4'.")

else:
    print("Invalid mode! Choose from 'image', 'video', or 'webcam'.")
    exit()