from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss
from sklearn.model_selection import train_test_split
import numpy as np
import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU ##alpha=0.1
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

import time


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

dataset = VOCdataset()
data,bboxes,labels,no_objects, imagePaths = dataset.load_dataset(imageDir,annotDir)

#default number of grids: no_grids =11 and default number of box predictions: B=2

B = 2 # number of bbox predictions per cell
no_grids = 7 # number of grid cells per image
GT = GridTransform(B, no_grids)

bboxes_grids,labels_grids = GT.transform(bboxes,labels,no_objects)

#print(data.shape)
#print(bboxes_grids.shape)
#print(labels_grids.shape)


one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
one_head_labels = np.reshape(one_head_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))

trainImages, valImages, trainLabels, valLabels, trainPaths, valPaths = train_test_split(data, one_head_labels, imagePaths, test_size = 0.15, random_state=42)



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

"""
model = ResNet50(include_top=False,weights='imagenet', input_shape=(224,224,3))

for layer in model.layers:
    layer.trainable=False

head = Conv2D(512,3,(1,1),padding='valid',kernel_regularizer='l2',kernel_initializer=tf.keras.initializers.HeUniform(),name='fc_1')(model.output)
head = LeakyReLU(alpha=0.2)(head)
head = tf.keras.layers.BatchNormalization()(head)
head = Dropout(0.5)(head)
head = Conv2D(512,3,(1,1),padding='valid',kernel_regularizer='l2',kernel_initializer=tf.keras.initializers.HeUniform(),name='fc_2')(head)
head = LeakyReLU(alpha=0.2)(head)
head = tf.keras.layers.BatchNormalization()(head)
head = Dropout(0.5)(head)
head = Flatten()(head)
head = Dense((len(classes)+B*5)*no_grids*no_grids, activation='sigmoid', name = 'out')(head)#(len(classes)+B*5)*no_grids*no_grids

detector = Model(inputs=model.input,outputs = head)
"""

detector = load_model('output/conv_resnet_warm.hdf5', custom_objects = {"YoloLoss":YoloLoss,"mAP":GT.mAP})

#for layer in detector.layers[:]:
#    layer.trainable=False

for layer in detector.layers[19:]:
    layer.trainable= True


for (i,layer) in enumerate(detector.layers[:]):
    print(i,layer.name,layer.trainable)



plot_model(detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


def step_decay(epoch):
    if epoch < 30:
        lr = 0.0001
    if epoch > 29: 
        lr = 0.00005
    if epoch > 55:
        lr = 0.00001
    return lr

fname = os.path.sep.join(["output/conv_resnet_tune.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", save_best_only=True, mode = "min", verbose=1)
tensorboardname = "Pascal-model-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(tensorboardname))
lrate = LearningRateScheduler(step_decay)


callbacks = [checkpoint,tensorboard]#,reduce_lr,early_stop]#,LearningRateScheduler(step_decay)

loss = YoloLoss()
loss.__name__ = "YoloLoss"
# initialize the optimizer, compile the model, and show the model
# summary

lr = 0.00001
opt = Adam(learning_rate=lr)
detector.compile(loss=loss, optimizer=opt, metrics=[GT.mAP])#'mean_squared_error'

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")

H = detector.fit(
	trainImages,trainLabels,
	validation_data =(valImages,valLabels),
	batch_size=8,
	epochs=250,
    callbacks = callbacks,
	verbose=1)

detector.save_weights('output/resnet_warm_22martie')

