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
	imagePaths = open('output/test.txt').read().strip().split("\n")


print("Load trained model")
model = load_model('output/7x7epoch250loss3p37.hdf5', custom_objects = {"YoloLoss":YoloLoss})
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

no_grids=7
B=2
GT = GridTransform(B,no_grids)
fps_inference = []
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
        fps_inference.append(1/(time.time() - start))
        #print(prediction.shape)
        prediction = np.reshape(prediction[0],(30,no_grids,no_grids))
        #print(prediction.shape)
        boxPred = prediction[20:30,...]       
        classPred = prediction[:20,...]
        #print(boxPred.shape)
        image1 = cv2.imread(imagePath)
        (h, w) = image1.shape[:2]
        #print(boxPred[0,...])
        #print(boxPred[5])
        output_path = 'images_pred/test_nms/image_{}.jpg'.format(step)
        #GT.transform_from_grid(boxPred,classPred,h,w,image1,output_path)
        GT.transform_with_nms(boxPred,classPred,h,w,image1,output_path)
        fps_inference.append(1/(time.time() - start))
print("Mean FPS value of inference is: {}".format(sum(fps_inference)/len(fps_inference)))    
        
        