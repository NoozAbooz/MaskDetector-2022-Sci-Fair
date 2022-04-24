# Haha CPU go brrrrrrrrrrrrrrrrrrrrr

'''
python train_mask_detector.py ^
--dataset Dataset --plot PlotV2.png --model MaskDetectorV2.model
'''

# Gotta import them all!
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import os

# Take arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", type=str, default="mask_detector2.model")
args = vars(ap.parse_args())

# Define main training paramaters
INIT_LR = 1e-4
EPOCHS = 25
BS = 10

# Get dict of dataset paths and init other needed dicts
print("Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Pre-process every image
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	# Resize image (224x224) and convert to array
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# Add array to dict
	data.append(image)
	labels.append(label)

# Convert array to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# One-hot encoding (https://stackoverflow.com/questions/60868391/how-to-view-class-labels-after-one-hot-encoding-during-training-testing-and-afte)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Use 80% of data for training and 20 for testing 9https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Set parms for random image movement (https://faroit.com/keras-docs/1.2.2/preprocessing/image/)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

# Load MobileNetV2 (https://pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Create model head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Put model head on model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze training layers so they won't change during first training round
for layer in baseModel.layers:
	layer.trainable = False

# Configure model for training with adam optmiser (https://keras.io/api/metrics/)
print("Compiling model with adam...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train head model
print("Training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Make predictions on the testing dataset
print("Testing model...")
predIdxs = model.predict(testX, batch_size=BS)

# Prediction index
predIdxs = np.argmax(predIdxs, axis=1)

# Show final report for results
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Save model to disk
print("Saving...")
model.save(args["model"], save_format="h5")
