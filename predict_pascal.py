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
import time
from tensorflow.keras.utils import plot_model
import tensorflow as tf

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

# Get paths to the test images
filetype = mimetypes.guess_type('output/train.txt')[0]

# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the image paths in our testing file
	imagePaths = open('output/val.txt').read().strip().split("\n")

no_grids=7
B=2
GT = GridTransform(B,no_grids)
fps_inference = []

print("Load trained model")
model = load_model('output/best_model_retry.hdf5', custom_objects = {"YoloLoss":YoloLoss,"mAP":GT.mAP})
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


for (step,imagePath) in enumerate(imagePaths):
    if step ==600:
        break
    else:
        start = time.time()
        
        # load the input image (in Keras format) from disk and preprocess
        # it, scaling the pixel intensities to the range [0, 1]
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        # predict the bounding box of the object along with the class label
        prediction = model.predict(image)
        
        prediction = np.reshape(prediction[0],(30,no_grids,no_grids))
        
        boxPred = prediction[20:30,...]
        classPred = prediction[:20,...]
        
        image1 = cv2.imread(imagePath)
        output_path = 'images_pred/val_nms/image_{}.jpg'.format(step)
        #image=GT.transform_from_grid(boxPred,classPred,image1)
        image_final = GT.transform_with_nms(boxPred,classPred,image1)
        #cv2.imshow('Image',image_final)
        cv2.imwrite(output_path,image_final)
        fps_inference.append(1/(time.time() - start))
print("Mean FPS value of inference is: {}".format(sum(fps_inference)/len(fps_inference)))    
        
        
