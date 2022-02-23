from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
#from tensorflow.keras.utils.vis_utils import plot_model

from tensorflow.keras.models import model_from_json



classes = ['airplanes','car', 'faces', 'faces_easy', 'motorbikes']

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
#model = load_model('/content/gdrive/MyDrive/IMSAR/multibox_tuned.h5', custom_objects = {"classification_loss": focal_loss,
#	"bounding_box": smooth_L1_loss})

 
# load json and create model
json_file = open('single_detector7grid.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("single_detector7grid.h5")
print("Loaded model from disk")

B=2
no_grids=7

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA)

    image = img_to_array(frame) / 255.0
    image = np.expand_dims(image, axis=0)

	# predict the bounding box of the object along with the class label
    (boxPreds,labelPreds) = model.predict(image)

	# Predicted labels.bboxes and the position of the object
    boxPreds = np.reshape(boxPreds[0],(B*4,no_grids,no_grids))
    labelPreds = np.reshape(labelPreds[0],(len(classes)+1,no_grids,no_grids))
    has_obj = labelPreds[len(classes)]
    # Groundtruth labels and bboxes
    #true_bboxes = np.reshape(valBBoxes[n],(B*4,no_grids,no_grids))
    #true_labels = np.reshape(valLabels[n],((len(classes)+1),no_grids,no_grids))

    #print(true_bboxes)
    #print(boxPreds)

    #print(true_labels)
    print(labelPreds)

    print(has_obj)
    #print(true_labels[len(lb.classes_)])

    # Get the position of the object and the confidence of the position
    max=0
    i=0
    j=0
    for l in range(no_grids):
        for k in range(no_grids):
            if has_obj[l][k] > max:
                max = has_obj[l][k]
                i = l
                j = k
    
    class_pos = np.argmax(labelPreds[:len(classes),i,j], axis=0)
    #print(lb.classes_)
    print("Confidence:{}".format(max))
    label = classes[class_pos]
    print(label)


    image = imutils.resize(frame, width=600)
    (h, w) = image.shape[:2]
    print(w,h)
    for k in range(B):
        # scale the predicted bounding box coordinates based on the image
        # dimensions
        cx_box = boxPreds[4*k+0][i][j]
        cy_box = boxPreds[4*k+1][i][j]
        width = boxPreds[4*k+2][i][j]
        height = boxPreds[4*k+3][i][j]

        cx = i/no_grids+cx_box/no_grids
        cy = j/no_grids+cy_box/no_grids

        startX = int((cx-width/2)*w)
        startY = int((cy-height/2)*h)
        endX = int((cx+width/2)*w)
        endY = int((cy+height/2)*h)
        print(startX,startY,endX,endY)

        # draw the predicted bounding box and class label on the image
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,	0.65, (0, 255, 0), 2)
    #cv2_imshow(image)
    cv2.imshow('Input', image)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
