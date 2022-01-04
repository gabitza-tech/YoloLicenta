import numpy as np
from keras.losses import MeanSquaredError
from data.data import VOCdataset
from data.transforms import GridTransform
from data.functions import convert_cellbox_to_boxes, intersection_over_union
from tensorflow import keras  
from keras import backend as K
import tensorflow as tf

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class YoloLoss:
        
    def __init__(self):
        self.mse = MeanSquaredError(reduction="sum")
        self.B = 2
        self.no_grids = 11
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.ious = []

    def __call__(self,y_true,y_pred):
        

        ## WORKS FOR MORE THAN ONE IMAGE IN THE BATCH
        prediction = np.reshape(y_pred,(-1,(len(classes)+self.B*5),self.no_grids,self.no_grids)) #dim (BATCH,30,6,6)
        target = np.reshape(y_true,(-1,(len(classes)+self.B*5),self.no_grids,self.no_grids)) #dim (BATCH,30,6,6)

        prediction_center = convert_cellbox_to_boxes(prediction) # dim (BATCH,10,6,6)
        target_center = convert_cellbox_to_boxes(target) #dim (BATCH,10,6,6)

        print(prediction_center.shape)
        print(target_center.shape)
        ### CALCULATE THE IOU FOR EACH BBOX PREDICTION
        for i in range(self.B):
            x = intersection_over_union(prediction_center[:,(0+i*5):(0+(i+1)*5),...],target_center[:,0:5,...])
            self.ious.append(x)
        
        self.ious = np.array(self.ious,dtype="float32")
        ious = np.concatenate([np.expand_dims(self.ious[0],axis=0),np.expand_dims(self.ious[1],axis=0)])
        
        ### FIND THE BEST BBOX ACCORDING TO THE HIGHEST IOU
        bestbox = np.argmax(ious,axis=0)
        bestbox = np.expand_dims(bestbox,axis=1) # dim: (BATCH,1,6,6)
        
        ### Iobj_i ---see if there is an object in a particular cell 
        exists_box = np.expand_dims(target_center[:,0,...],axis=1) # dim (BATCH,1,6,6)
        
        """ 
        BOXES COORDINATES LOSS
        """   
        box_predictions = exists_box * (
            bestbox * prediction_center[:,6:10,...]
            + (1-bestbox) * prediction_center[:,1:5,...]
        ) ## box predictions dim: (BATCH,4,6,6)

        box_targets = exists_box * target_center[:,1:5,...] # dim:(BATCH,4,6,6)

        box_loss = self.mse(box_predictions,box_targets)
        print(box_loss)

        """
        OBJECT LOSS
        """
        ## pred_box represents the confidence score for the box with the highest IOU
        pred_box = (bestbox * np.expand_dims(prediction_center[:,5,...],axis=1)
                    + (1-bestbox)*np.expand_dims(prediction_center[:,0,...],axis=1))
        
        object_loss = self.mse(
            (exists_box * pred_box),
            (exists_box * np.expand_dims(target_center[:,0,...],axis=1))
            )
        print(object_loss)

        """
        NO OBJECT LOSS
        """
        #x = np.reshape(((1-exists_box)*np.expand_dims(prediction_center[:,0,...],axis=1)),(-1,1,36))
        #print(x.shape)
        no_object_loss = self.mse(np.reshape(((1-exists_box)*np.expand_dims(prediction_center[:,0,...],axis=1)),(-1,1,self.no_grids*self.no_grids)),
                                    np.reshape(np.expand_dims(target_center[:,0,...],axis=1),(-1,1,self.no_grids*self.no_grids)))
        no_object_loss += self.mse(np.reshape(((1-exists_box)*np.expand_dims(prediction_center[:,5,...],axis=1)),(-1,1,self.no_grids*self.no_grids)),
                                    np.reshape(np.expand_dims(target_center[:,0,...],axis=1),(-1,1,self.no_grids*self.no_grids)))
        
        ################
        # DE TINUT MINTE DE MODIFICAT AL DOILEA NO_OBJECT_LOSS CAND VREAU SA FOLOSESC
        # AMBELE PREDICTION BOXES SA MODIF TARGET cENTER IN target_center[:,0,...]
        #################    
        print(no_object_loss)

        """
        CLASS LOSS
        """
        #print(exists_box*np.expand_dims(prediction_center[:,:20,...],axis=1))
        #class_loss = self.mse(
        #        np.reshape(exists_box*np.expand_dims(prediction[:,:20,...],axis=1),(-1,20,36)),
        #        np.reshape(exists_box*np.expand_dims(target[:,:20,...],axis=1),(-1,20,36))
        #)
        class_loss = self.mse(
                (exists_box*np.expand_dims(prediction[:,:20,...],axis=1)),
                (exists_box*np.expand_dims(target[:,:20,...],axis=1))
        )
        
        print("Class loss:{}".format(class_loss))
        
        """
        FINAL LOSS
        """

        loss = tf.add(tf.add(
            self.lambda_coord * box_loss  
            ,object_loss),tf.add(  
            self.lambda_noobj * no_object_loss  
            ,class_loss)
        )
        #loss=tf.convert_to_tensor(loss,dtype=tf.float32)
        print("Final loss:{}".format(loss))
        return loss

        
""""
### TESTING
imageDir = 'dataset/images'
annotDir = 'dataset/annotations'

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

one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
#print(one_head_labels.shape)
one_head_labels = np.reshape(one_head_labels,(one_head_labels.shape[0],(len(classes)+B*5)*no_grids*no_grids))
#print(one_head_labels.shape)

loss = YoloLoss()

target_labels = np.copy(one_head_labels[0:16])
target_labels = np.reshape(target_labels,(-1,(len(classes)+B*5),no_grids,no_grids))

target_labels = np.reshape(target_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))

loss = loss(one_head_labels[0:16],target_labels)
print(loss)
"""