# Small disclaimer
print("[INFO] Starting program... (might take a while to load)")

# Import stuff
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import threading
import time
import cv2
import os
import paramiko

# Define important variables
faceDetector = "Face Detector"
model = "MaskDetectorV2.model"
minConfidence = 0.7
timeoutPeriod = 5
disableEV3 = True

# Init vars for FPS counter
start_time = time.time()
display_time = 2
fc = 0
FPS = 0

# Establish EV3 connection via SSH
host = 'ev3dev.local'
username = 'robot'
password = 'pihole'
con = paramiko.SSHClient()
con.set_missing_host_key_policy(paramiko.AutoAddPolicy())
if disableEV3 == False:
    # Establish SSH connection
    con.connect(hostname=host, username=username, password=password)
    print("[INFO] Connected to EV3")
inTimeout = False

''' Load models '''

# Load the face detector model (https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN)
print("[INFO] Loading face detector model...")
prototxtPath = faceDetector + "\\deploy.prototxt"
weightsPath = faceDetector + "\\facedetector.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model
print("[INFO] Loading mask model...")
maskNet = load_model(model)

''' Define main functions '''

def EV3Timeout():
    global inTimeout
    inTimeout = True
    time.sleep(timeoutPeriod)
    inTimeout = False

def EV3Caller(maskless):
    # Send mask total count to EV3
    if not inTimeout:
        if disableEV3 == False:
            print("[INFO] Calling EV3 Script")
            timer = threading.Thread(target=EV3Timeout)
            con.exec_command('~/EV3Code/DispenseV2.sh ' + str(maskless))
            timer.start()

def FaceDetector(frame, faceNet):
    global detections	

    # Grab webcam size and then construct a blob (group of connected pixels in an image that share some common property https://learnopencv.com/blob-detection-using-opencv-python-c/)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Pass the blob through face detector
    faceNet.setInput(blob)
    detections = faceNet.forward()

# Main mask detection function
def MaskDetector(frame, maskNet):
    (h, w) = frame.shape[:2]
    FaceDetector(frame, faceNet)

    # Init list of faces, their corresponding locations, and the list of predictions from face detector
    faces = []
    locs = []
    preds = []

    # Loop over each face detection
    for i in range(0, detections.shape[2]):
        
        # Extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > minConfidence:
            
            # Get the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fit inside the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face return value, convert it from BGR to RGB ordering, resize to 244x244 and process it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # Add the face and bounding boxes to their respective dicts
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make a mask prediction if at least one face was detected
    if len(faces) > 0:
        # Process all faces instead of one at a time for speeeed
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=20)

    # Return face locations and their corresponding locations
    return (locs, preds)

# Init the video stream
print("[INFO] Ininializing video...")
vs = VideoStream(src=0).start()

# Loop over each frame from video
while True:
    # Resize to 600x400
    frame = vs.read()
    frame = imutils.resize(frame, width=600, height=400)

    # Run the faces through face/mask detector and log return value
    (locs, preds) = MaskDetector(frame, maskNet)

    # Calculate FPS
    fc+=1
    TIME = time.time() - start_time

    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()

    fps_disp = "FPS: "+str(FPS)[:5]

    # Loop over each face
    maskless = 0
    for (box, pred) in zip(locs, preds):
        
        # Unpack bounding box positions and labels
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Set colours for boxes
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Add to count of people not wearing masks
        if label == "No Mask":
            maskless += 1

        # Add confidence to bounding box
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Render boxes on faces
        cv2.putText(frame, label, (startX, endY + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    print("People not wearing masks: " + str(maskless))

    # Draw FPS on window
    cv2.putText(frame, fps_disp, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 217, 67), 2)

    # Show final frame with the boxes
    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Call the EV3 code if 1 or more people aren't wearing masks and reset counter
    if maskless > 0:
        EV3Caller(maskless)
        maskless = 0
    
    # Close window if `Q` is pressed
    if key == ord("q"):
        break

# Close window and stop webcam
cv2.destroyAllWindows()
vs.stop()
