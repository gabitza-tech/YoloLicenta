from data.yolo_loss import yolo_loss
from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss,class_loss,box_loss,obj_loss,noobj_loss


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

import numpy as np
import mimetypes
import argparse
import pickle
import cv2
import os
import time

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
#model = load_model('output/x.hdf5', custom_objects = {"yolo_loss":yolo_loss,"mAP":GT.mAP})
model = load_model('output/7x7epoch250loss3p37.hdf5', custom_objects = {"YoloLoss":YoloLoss})
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""
When using the test.txt file: uncomment lines 49,51,58 and comment line 50.
"""
#testPaths = 'dataset/testImages'
for (step,imagePath) in enumerate(imagePaths):
#for (step, imagePath) in enumerate(os.listdir(testPaths)):
    if step ==200:
        break
        #continue
    else:
        start = time.time()
        
        #imagePath = os.path.join(testPaths,imagePath) 
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)/255
	#Preprocess the images before feeding them to the network as they were preprocessed when training
        #image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        
        # predict the bounding box of the object along with the class label
        prediction = model.predict(image)
        
        prediction = np.reshape(prediction[0],((len(classes)+B*5),no_grids*no_grids))
        
        boxPred = prediction[len(classes):len(classes)+B*5,...]
        classPred = prediction[:len(classes),...]

        image_test = cv2.imread(imagePath)
        #Choose which type of visualization you want: without or with nms
        #image_final=GT.transform_from_grid(boxPred,classPred,image_test)

        image_final = GT.transform_with_nms(boxPred,classPred,image_test)

        output_path = 'images_pred/test_nms/image_{}.jpg'.format(step)

        #cv2.imshow('Image',image_final)
        cv2.imwrite(output_path,image_final)
        fps_inference.append(1/(time.time() - start))

print("Mean FPS value of inference is: {}".format(sum(fps_inference)/len(fps_inference)))    
        
        
