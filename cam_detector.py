from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss
from data.yolo_loss import yolo_loss
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
#from tensorflow.keras.utils.vis_utils import plot_model

from tensorflow.keras.models import model_from_json



classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

# load our object detector and label binarizer from disk

 
print("Load trained model")
#put you own model here with your custom objects. If trained with yolo_loss and mAP function, add them in custom objects
model = load_model('output/7x7epoch250loss3p37.hdf5', custom_objects = {"YoloLoss":YoloLoss})

no_grids=7
B=2
GT = GridTransform(B,no_grids)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA) 
    image_p = img_to_array(image) / 255.0
    image_p = np.expand_dims(image, axis=0)

    # predict the bounding box of the object along with the class label
    prediction = model.predict(image_p)
    #print(prediction.shape)
    prediction = np.reshape(prediction[0],((len(classes)+B*5),no_grids*no_grids))
    #print(prediction.shape)

    boxPred = prediction[len(classes):(len(classes)+B*5),...]       
    classPred = prediction[:len(classes),...]

    (h, w) = frame.shape[:2]
    
    image = GT.transform_with_nms(boxPred,classPred,image)
    cv2.imshow('Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
