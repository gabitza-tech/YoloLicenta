from data.data import VOCdataset
from data.transforms import GridTransform
#from data.losses import YoloLoss
from data.losses import YoloLoss
from sklearn.model_selection import train_test_split
import numpy as np
import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU ##alpha=0.1
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error
#tf.enable_eager_execution()
#tf.compat.v1.enable_eager_execution()
#print(tf.executing_eagerly())

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
no_grids = 9 # number of grid cells per image
GT = GridTransform(B, no_grids)

bboxes_grids,labels_grids = GT.transform(data,bboxes,labels,no_objects)

#print(data.shape)
#print(bboxes_grids.shape)
#print(labels_grids.shape)

##############################
## FOR TWO HEADS NETOWRK SPLIT

#bboxes_grids = np.reshape(bboxes_grids,(bboxes_grids.shape[0],bboxes_grids.shape[1]*bboxes_grids.shape[2]*bboxes_grids.shape[3]))
#labels_grids = np.reshape(labels_grids,(labels_grids.shape[0],labels_grids.shape[1]*labels_grids.shape[2]*labels_grids.shape[3]))

#print(bboxes_grids.shape)
#print(labels_grids.shape)


#[trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes, trainPaths, testPaths] = train_test_split(data, labels_grids, bboxes_grids, imagePaths, test_size=0.10, random_state=42)

#[trainImages, valImages, trainLabels, valLabels, trainBBoxes, valBBoxes, trainPaths, valPaths] = train_test_split(trainImages, trainLabels, trainBBoxes, trainPaths, test_size=0.25, random_state=42)

#print(trainImages.shape)
#print(valImages.shape)
#print(testImages.shape)
###############################

###############################
## IF I WANT TO USE ONLY ONE HEAD IN THE NETWORK

one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
#print(one_head_labels.shape)
#print(one_head_labels[1][20])
#print(one_head_labels[1][21:25])
#print(one_head_labels[1][0])
#print(one_head_labels[1][7])
one_head_labels = np.reshape(one_head_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))
#print(one_head_labels.shape)
#print(bboxes[1])

trainImages, testImages, trainLabels, testLabels, trainPaths, testPaths = train_test_split(data, one_head_labels, imagePaths, test_size = 0.1, random_state=42)
trainImages, valImages, trainLabels, valLabels = train_test_split(trainImages, trainLabels, test_size = 0.2, random_state=42)

#trainImagesdataset = tf.data.Dataset.from_tensor_slices(
#    (trainImages,trainLabels)
#)

#trainImagesdataset.batch(batch_size=32)



#valImagesdataset = tf.data.Dataset.from_tensor_slices(
#    (valImages,valLabels)
#)
#valImagesdataset.batch(batch_size=32)

#testImagesdataset = tf.data.Dataset.from_tensor_slices(
#    (testImages,testLabels)
#)
#testImagesdataset.batch(batch_size=32)

#print(trainLabels.shape)
#print(valLabels.shape)
#print(testLabels.shape)

###############################

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open("output/test.txt", "w")
f.write("\n".join(testPaths))
f.close()

model = ResNet50(include_top=False,weights='imagenet', input_shape=(224,224,3))
flatten = Flatten()(model.output)

head = Dense(2048,name='fc_1',activation = 'elu')(flatten)
head = BatchNormalization(name='batchbox1')(head)
head = Dropout(0.5)(head)
head = Dense(1024,name='fc_2',activation = 'elu')(head)
head = BatchNormalization(name='batchbox2')(head)
head = Dropout(0.5)(head)
head = Dense((len(classes)+B*5)*no_grids*no_grids, activation = 'sigmoid', name = 'fc_3')(head)#(len(classes)+B*5)*no_grids*no_grids

detector = Model(inputs=model.input,outputs = head)
plot_model(detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.001
    factor = 0.5
    dropEvery = 20

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)

fname = os.path.sep.join(["output/weights-best.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", save_best_only=True, mode = "min", verbose=1)

callbacks = [LearningRateScheduler(step_decay),checkpoint]

loss = YoloLoss()
#loss = MeanSquaredError(reduction="sum")
loss.__name__ = "YoloLoss"
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=0.001)
detector.compile(loss=loss, optimizer=opt, metrics=["mae"])#'mean_squared_error'

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")

H = detector.fit(
	trainImages,trainLabels,
	validation_data =(valImages,valLabels),
	batch_size=16,
	epochs=80,
    callbacks = callbacks,
	verbose=1)

#loss = loss.forward(y_pred, y_true)