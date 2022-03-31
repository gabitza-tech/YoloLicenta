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
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

import time

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})


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

trainImages, testImages, trainLabels, testLabels, trainPaths, testPaths = train_test_split(data, one_head_labels, imagePaths, test_size = 0.1, random_state=42)
trainImages, valImages, trainLabels, valLabels,trainPaths,valPaths = train_test_split(trainImages, trainLabels,trainPaths, test_size = 0.2, random_state=42)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open("output/test.txt", "w")
f.write("\n".join(testPaths))
f.close()


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
  
model = ResNet50(include_top=False,weights='imagenet', input_shape=(224,224,3))

for layer in model.layers:
    layer.trainable=False

flatten = Flatten()(model.output)
head = Dense(2048,name='fc_1',activation='Mish')(flatten)
#head = LeakyReLU(alpha=0.1)(head)
head = Dropout(0.6)(head)
head = Dense((len(classes)+B*5)*no_grids*no_grids, activation = 'sigmoid', name = 'fc_3')(head)#(len(classes)+B*5)*no_grids*no_grids

detector = Model(inputs=model.input,outputs = head)
"""
detector = load_model('output/7x7epoch250loss3p37.hdf5', custom_objects = {"YoloLoss":YoloLoss})

for layer in detector.layers[:]:
    layer.trainable=True

#for layer in detector.layers[-36:]:
#    layer.trainable=True
   """
for layer in detector.layers[:]:
    print(layer.name,layer.trainable)
  
plot_model(detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

fname = os.path.sep.join(["output/warm.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", save_best_only=True, mode = "min", verbose=1)
tensorboardname = "Pascal-model-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(tensorboardname))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,mode = 'min',
                              patience=6, min_lr=0.0000001,verbose=1)
early_stop = EarlyStopping(monitor='val_loss',patience = 10, min_delta =0.1, mode= 'min')

callbacks = [checkpoint,tensorboard,reduce_lr,early_stop]#,LearningRateScheduler(step_decay)

loss = YoloLoss()
loss.__name__ = "YoloLoss"
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=0.0001)
detector.compile(loss=loss, optimizer=opt, metrics=["mae"])#'mean_squared_error'

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")

H = detector.fit(
	trainImages,trainLabels,
	validation_data =(valImages,valLabels),
	batch_size=16,
	epochs=100,
    callbacks = callbacks,
	verbose=1)

detector.save_weights('output/resnet_warm_22martie')

# serialize model to JSON
#model_json = detector.to_json()
#with open("output/experiment.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#detector.save_weights("output/experiment.h5")
#print("Saved model to disk")
