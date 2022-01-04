from data.data import VOCdataset
from data.transforms import GridTransform
from data.losses import YoloLoss
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU ##alpha=0.1
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model, Model
from keras import backend as K
from keras.applications.resnet import ResNet50
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

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
no_grids = 11 # number of grid cells per image
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
print(one_head_labels.shape)
#one_head_labels = np.reshape(one_head_labels,(one_head_labels.shape[0],(len(classes)+B*5)*no_grids*no_grids))
#print(one_head_labels.shape)
#print(bboxes[1])

trainImages, testImages, trainLabels, testLabels, trainPaths, testPaths = train_test_split(data, one_head_labels, imagePaths, test_size = 0.1, random_state=42)
trainImages, valImages, trainLabels, valLabels = train_test_split(trainImages, trainLabels, test_size = 0.2, random_state=42)

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

head = Dense(512,name='dense_box1',activation = 'relu')(flatten)
head = BatchNormalization(name='batchbox1')(head)
head = Dropout(0.5)(head)
head = Dense(256,name='dense_box2',activation = 'relu')(head)
head = BatchNormalization(name='batchbox2')(head)
head = Dropout(0.5)(head)
head = Dense(128,name='dense_box3',activation = 'relu')(head)
head = BatchNormalization(name='batchbox3')(head)
head = Dropout(0.5)(head)
head = Dense((len(classes)+B*5)*no_grids*no_grids, activation='softmax', name = 'bounding_box')(head)

detector = Model(inputs=model.input,outputs = head)
plot_model(detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.001
    factor = 0.5
    dropEvery = 8

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#fname = os.path.sep.join([logdir,"weights-best.hdf5"])
#checkpoint = ModelCheckpoint(fname, monitor="val_loss", save_best_only=True, mode = "min", verbose=1)

callbacks = [LearningRateScheduler(step_decay)]

loss = YoloLoss()

loss.__name__ = "YoloLoss"
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=0.001)
detector.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
H = detector.fit(
	trainImages, trainLabels,
	validation_data=(valImages, valLabels),
	batch_size=16,
	epochs=40,
    callbacks = callbacks,
	verbose=1)

#loss = loss.forward(y_pred, y_true)