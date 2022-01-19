# Mask detector 5000 - Michael Zheng 2022

# Import stuff
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import threading
import time
import cv2
import os
import paramiko

# Define vars and functions for EV3 Code
host = 'ev3dev.local'
username = 'robot'
password = 'pihole'
con = paramiko.SSHClient()
con.set_missing_host_key_policy(paramiko.AutoAddPolicy())
con.connect(host, username=username, password=password)
inTimeout = False

def timeout(secs):
    global inTimeout
    inTimeout = True
    time.sleep(secs)
    inTimeout = False

def call_ev3_script(no_mask_count):
    # Send mask total count to EV3
    if not inTimeout:
        print("[INFO] Calling EV3 Script")
        timer = threading.Thread(target=timeout, args=(5,))
        con.exec_command('~/EV3Code/DispenseV2.sh ' + str(no_mask_count))
        timer.start()

# Mask detection function
def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grab frame dimensions and then construct a blob with it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Pass the blob through network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Init list of faces, their corresponding locations, and the list of predictions from face detector
	faces = []
	locs = []
	preds = []

	# Loop over each face detection
	for i in range(0, detections.shape[2]):
		
		# Extract the confidence associated with the detection
		confidence = detections[0, 0, i, 2]

		# Filter out weak detections by ensuring the confidence is greater than the minimum
		if confidence > args["confidence"]:
			
			# Compute the (x, y) coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="FaceDetector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="MaskDetectorV2.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	no_mask_count = 0
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# send singnal to EV3 when label is 'Mask'
		if label == "No Mask":
			no_mask_count = no_mask_count + 1

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	#print(no_mask_count)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# call EV3 when no_mask_count is over 0
	if no_mask_count > 0:
		call_ev3_script(no_mask_count)
		no_mask_count = 0
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do cleanup
cv2.destroyAllWindows()
vs.stop()
