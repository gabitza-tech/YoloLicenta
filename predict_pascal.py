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

# determine the input file type, but assume that we're working with
# single input image
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
#for (n,imagePath) in enumerate(imagePaths):
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
image = load_img(imagePaths[7], target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# predict the bounding box of the object along with the class label
prediction = model.predict(image)
print(prediction.shape)
prediction = np.reshape(prediction,(-1,30,no_grids,no_grids))
print(prediction.shape)

boxPred1 = prediction[0,21:25,...]
boxPred2 = prediction[0,26:30,...]
confPred1 = prediction[:,20,...]
confPred2 = prediction[:,25,...]
classPreds = prediction[0,:20,...]

#print(boxPred1.shape)
print(confPred1)
print(confPred2)
#print(classPreds.shape)
#print(prediction[0,21:24,...])
#print(prediction[0,26:30,...])

image1 = cv2.imread(imagePaths[7])
image1 = imutils.resize(image1, width=600)
(h, w) = image1.shape[:2]

for (i,n) in enumerate(confPred2[0]):
    for (j,m) in enumerate(n):
        if m>0.3:
            labelpos = np.argmax(classPreds[:20,i,j],axis=0)
            label = classes[labelpos]

            for k in range(B):
                # scale the predicted bounding box coordinates based on the image
                # dimensions
                cx_box = boxPred2[0][i][j]
                cy_box = boxPred2[1][i][j]
                width = boxPred2[2][i][j]
                height = boxPred2[3][i][j]

                cx = i/no_grids+cx_box/no_grids
                cy = j/no_grids+cy_box/no_grids

                startX = int((cx-width/2)*w)
                startY = int((cy-height/2)*h)
                endX = int((cx+width/2)*w)
                endY = int((cy+height/2)*h)
                #print(startX,startY,endX,endY)

                # draw the predicted bounding box and class label on the image
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(image1, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
            cv2.putText(image1, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,	0.65, (0, 255, 0), 2)
cv2.imshow('ya',image1)
cv2.waitKey(0) 
cv2.destroyAllWindows() 