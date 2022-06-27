# Functions for data loading, transformations, metrics and the loss function
from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss,class_loss,box_loss,obj_loss,noobj_loss
from data.yolo_loss import yolo_loss

#functions for OneCyclePolicy
from data.callback import SGDR
from CLR.clr_callback import CyclicLR
from clr import LRFinder

import time
import tempfile
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

"""
Pretrained networks
"""
from tensorflow.keras.applications.resnet import ResNet50
#from tensorflow.keras.applications import ResNet50V2
#from tensorflow.keras.applications import InceptionV3
#from tensorflow.keras.applications import MobileNetV2
#from tensorflow.keras.applications import EfficientNetB0

"""
Preprcessing
"""
#from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.efficientnet import preprocess_input

"""
Callbacks and plotter
"""
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping

imageDir = 'dataset/images'
annotDir = 'dataset/annotations'
classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

data = []
bboxes = []
labels = []
no_objects = []
imagePaths = []
"""
Loading data from PascalVOC2012 dataset
"""
dataset = VOCdataset(224,224)
data,bboxes,labels,no_objects, imagePaths = dataset.load_dataset(imageDir,annotDir)


B = 2 # number of bbox predictions per cell
no_grids = 7 # number of grid cells per image
GT = GridTransform(B, no_grids)

"""
Transform from Pascal format to YOLO format
"""
bboxes_grids,labels_grids = GT.transform(bboxes,labels,no_objects)

#print(data.shape)
#print(bboxes_grids.shape)
#print(labels_grids.shape)

data = preprocess_input(data) # keras preprocessing for pretrained model

"""
Labels flattening and splitting train-val dataset
"""
one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
one_head_labels = np.reshape(one_head_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))
trainImages, valImages, trainLabels, valLabels, trainPaths, valPaths = train_test_split(data, one_head_labels, imagePaths, test_size = 0.1, random_state=42)


# write the training image paths to disk so that we can use then
# when evaluating/testing our object detector on training set
print("[INFO] saving training image paths...")
f = open("output/train.txt", "w")
f.write("\n".join(trainPaths))
f.close()

# when evaluating/testing our object detector on training set
print("[INFO] saving val image paths...")
f = open("output/val.txt", "w")
f.write("\n".join(valPaths))
f.close()

# ADD L2 REGULARIZATION TO CONV LAYERS IN THE PRETRAINED NETWORK
weight_decay= 3e-4
def add_regularization(model, regularizer=tf.keras.regularizers.l2(weight_decay)):
    
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model


    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)


    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

#model = MobileNetV2(include_top=False,weights='imagenet',input_shape=(224,224,3))
model = ResNet50(include_top=False,weights='imagenet', input_shape=(224,224,3))



model = add_regularization(model)

for layer in model.layers[:]:
    layer.trainable=True

head = Flatten()(model.output)
head = Dense(1024, name = 'fc1')(head)
head = LeakyReLU(0.1)(head)
head = Dropout(0.3)(head)
head = Dense(30*no_grids*no_grids,activation = 'sigmoid', name='out')(head)

detector = Model(inputs=model.input,outputs = head)

"""

detector = load_model('output/mobile_over.hdf5', custom_objects = {"yolo_loss":yolo_loss,"mAP":GT.mAP})#,"class_loss":class_loss,"box_loss":box_loss,"obj_loss":obj_loss,"noobj_loss":noobj_loss})
"""

for (i,layer) in enumerate(detector.layers[:]):
    print(i,layer.name,layer.trainable)

plot_model(detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

BATCH_SIZE = 16
num_epoch = 200

class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

fname = os.path.sep.join(["output/inception.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="loss", save_best_only=True, mode = "min", verbose=1)

tensorboardname = "Pascal-model-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(tensorboardname))


#lrate = LearningRateScheduler(warm,verbose=1) # Create a LRscheduler for warm-up and then training

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_delta=0.001,patience=3,verbose=1)

early_stop= EarlyStopping(monitor='loss',min_delta=0.001,mode='min',patience = 7)


callbacks = [checkpoint,LRTensorBoard(log_dir="logs/{}".format(tensorboardname))]#Choose the right callbacks for training

"""
Either choose the yolo_loss.py function from the yolo_loss.py or the YoloLoss() from the losses.py file
Recommended: yolo_loss.py
"""

#loss = YoloLoss()
#loss.__name__ = "YoloLoss"
# initialize the optimizer, compile the model, and show the model
# summary

lr = 1e-3# primu 9e-5
#opt = Adam(learning_rate=lr)
opt = SGD(lr=lr,momentum=0.9,nesterov= True)
detector.compile(loss=yolo_loss, optimizer=opt, metrics=[GT.mAP]) #Otherwise, use loss=loss if using the loss from losses.py

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")

H = detector.fit(
    trainImages,trainLabels,
    validation_data =(valImages,valLabels),
    batch_size=BATCH_SIZE,
    epochs = num_epoch,
    callbacks=callbacks,
    shuffle=True,
    verbose=1)
