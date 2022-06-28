#!/usr/bin/env python3

import rospy 
import matplotlib.pyplot as pyplot
import numpy as np 
import sys

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from data.transforms import GridTransform
from data.losses import YoloLoss
from data.yolo_loss import yolo_loss
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

import cv2

no_grids=7
B=2
GT = GridTransform(B,no_grids)


model = load_model('output/inception.hdf5', custom_objects = {"yolo_loss":yolo_loss,"mAP":GT.mAP})

bridge = CvBridge()
cnt = 0


def imgmsg_to_cv2(msg):
    #print(img_msg)
    #if img_msg.encoding != "rgb8":
    #    rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    #dtype = np.dtype("uint8") # Hardcode to 8 bits...
    #dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    #image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
    #                dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    #if img_msg.is_bigendian == (sys.byteorder == 'little'):
    #    image_opencv = image_opencv.byteswap().newbyteorder()
    
    #cv2.imwrite("saved_images/save.jpg",image_opencv)
    #image = cv2.resize(image_opencv,(224,224), interpolation=cv2.INTER_AREA)
    #image = img_to_array(image) / 255.0
    #image = np.expand_dims(image, axis=0)
    global cnt
    # Convert your ROS Image message to OpenCV2
    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    (h_init,w_init) = cv2_img.shape[:2]
    image = cv2.resize(cv2_img, (224,224), interpolation = cv2.INTER_AREA)
    image = img_to_array(image) 
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    # predict the bounding box of the object along with the class label
    prediction = model.predict(image)
    #print(prediction.shape)
    print("predicted")
    cnt = cnt +1 
    prediction = np.reshape(prediction[0],(30,no_grids*no_grids))
    #print(prediction.shape)

    boxPred = prediction[20:30,...]       
    classPred = prediction[:20,...]
    #print(boxPred.shape)
    #print(boxPred[0,...])
    #print(boxPred[5])
    image_final = GT.transform_with_nms(boxPred,classPred,cv2_img)

    image_to_ros = cv2.resize(image_final,(w_init,h_init), interpolation = cv2.INTER_AREA)
    ret_msg = bridge.cv2_to_imgmsg(image_to_ros,encoding="bgr8")

    pubDetections.publish(ret_msg)
    # Save your OpenCV2 image as a jpeg
    #cv2.imwrite('tiago_images/camera_new_image_{}.jpeg'.format(cnt), image_final)
    


if __name__ == '__main__':

    rospy.init_node('image_listener')

    # Define your image topic
    #image_topic = "/slower_image_raw"
    image_topic = "/xtion/rgb/image_raw"

    # Set up publishers
    #pubFaceFound = rospy.Publisher('/face_found', String, queue_size = 10)
    pubDetections = rospy.Publisher('/yolo_detections', Image, queue_size = 10)
    
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, imgmsg_to_cv2)

    # Spin until ctrl + c
    rospy.spin()
