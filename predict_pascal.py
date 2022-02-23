from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
from tensorflow.keras.utils import plot_model

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

# Get paths to the test images
filetype = mimetypes.guess_type('output/test.txt')[0]
imagePaths = ['output/test.txt']
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the image paths in our testing file
	imagePaths = open('output/test.txt').read().strip().split("\n")


print("Load trained model")
model = load_model('output/experiment3.hdf5', custom_objects = {"YoloLoss":YoloLoss})
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

no_grids=9
B=2
GT = GridTransform(B,no_grids)

for (step,imagePath) in enumerate(imagePaths):
    if step ==10:
        break
    else:
        # load the input image (in Keras format) from disk and preprocess
        # it, scaling the pixel intensities to the range [0, 1]
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # predict the bounding box of the object along with the class label
        prediction = model.predict(image)
        print(prediction.shape)
        prediction = np.reshape(prediction[0],(30,no_grids,no_grids))
        print(prediction.shape)

        boxPred = prediction[20:30,...]       
        classPred = prediction[:20,...]

        image1 = cv2.imread(imagePaths)
        image1 = imutils.resize(image1, width=600)
        (h, w) = image1.shape[:2]
        GT.transform_from_grid(boxPred,classPred,h,w,image)
