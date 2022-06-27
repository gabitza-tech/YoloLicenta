from imutils import paths
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from numpy import array

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

class VOCdataset:
    def __init__(self,w,h):            
        self.bboxes = []
        self.data = []
        self.no_objects = []
        self.labels = []
        self.imagePaths = []
        self.break_count = 0
        self.width_image = w
        self.height_image = h


    def read_content(self,xml_file:str):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        list_with_all_boxes = []

        for boxes in root.iter('object'):
            filename = root.find('filename').text
            label_obj = boxes.find("name").text
            xmin = int(float(boxes.find("bndbox/xmin").text))
            ymin = int(float(boxes.find("bndbox/ymin").text))
            xmax = int(float(boxes.find("bndbox/xmax").text))
            ymax = int(float(boxes.find("bndbox/ymax").text))
            width_image= int(float(root.find("size/width").text))
            height_image= int(float(root.find("size/height").text))
            one_bbox_and_label = [label_obj,xmin,ymin,xmax,ymax,width_image,height_image]
            list_with_all_boxes.append(one_bbox_and_label)
        return filename, list_with_all_boxes
    
    def center_and_hw(self,xmin,ymin,xmax,ymax):
        center_x = int((xmax+xmin)/2)
        center_y = int((ymax+ymin)/2)
        width = int((xmax-xmin))
        height = int((ymax-ymin))
        return center_x, center_y, width, height

    def load_dataset(self, imageDir, annotDir):
        for file in sorted(os.listdir(annotDir)):

            bboxes_per_image = []
            labels_per_image = []
            no_objects_per_image = 0
            annotPath = os.path.join(annotDir,file) 
            imageName, bboxes_image = self.read_content(annotPath) # READ CONTENT OF THE XML FILE
            imagePath = os.path.join(imageDir,imageName) # GET IMAGE PATH
            

            #image = cv2.imread(imagePath)
            #(h1,w1) = image.shape[:2]
            # get the label for each box in an image and the scaled coordinates between [0,1] of the bboxes in an image 
            for bbox in bboxes_image:
                w_image=bbox[5]
                h_image=bbox[6]
                label_name = bbox[0]
                label_bbox = classes.index(label_name)
                center_x,center_y, width,height = self.center_and_hw(bbox[1],bbox[2],bbox[3],bbox[4])
                center_x,center_y, width,height = round(center_x/w_image,3),round(center_y/h_image,3), round(width/w_image,3),round(height/h_image,3) ## we normalise the coordinates of the bboxes
                bboxes_per_image.append([center_x,center_y, width,height])
                labels_per_image.append(label_bbox)
                no_objects_per_image = no_objects_per_image + 1

                #startY = bbox[2] - 10 if bbox[2]>10 else bbox[2] + 10
                #cv2.putText(image, label_name,(bbox[1],startY), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                #cv2.rectangle(image,(bbox[1],bbox[2]),(bbox[3],bbox[4]),(0,255,0),1)
     
            #cv2.imshow('',image)
            #cv2.waitKey(0)

            image = load_img(imagePath, target_size=(self.width_image, self.height_image))
            image = img_to_array(image)

            self.bboxes.append(bboxes_per_image)
            self.labels.append(labels_per_image)
            self.no_objects.append(no_objects_per_image)
            self.data.append(image)
            self.imagePaths.append(imagePath)
             

            # to limit the amount of images read from the dataset
            self.break_count = self.break_count+1
            if self.break_count%100 == 0:
                print("INFO processed {}/17100 images".format(self.break_count))
            #if self.break_count == 100:
                #break

        # We want to calculate the maximum number of objects in an image so all the 
        # entries in the network have the same shape in order to transform them in numpy arrays

        max = 0
        for i in self.no_objects:
            if i > max:
                max = i
        print("Maximum number of objects in an image: {}".format(max))

        self.data = np.array(self.data,dtype="float32")
        self.no_objects = np.array(self.no_objects)

        
        return self.data,self.bboxes,self.labels,self.no_objects, self.imagePaths
""""

imageDir = 'dataset/images'
annotDir = 'dataset/annotations'

data = []
bboxes = []
labels = []
no_objects = []

dataset = VOCdataset(224,224)
data,bboxes,labels,no_objects,imagePaths = dataset.load_dataset(imageDir,annotDir)

#print(data.shape)
#print(no_objects.shape)
#print(labels)
print(bboxes[1])
"""
