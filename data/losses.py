import numpy as np
#from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error as mse
from data.data import VOCdataset
from data.transforms import GridTransform
from keras import backend as K
import tensorflow as tf

#tf.compat.v1.enable_eager_execution()

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class YoloLoss:
        
    def __init__(self):
        #self.mse = MeanSquaredError(reduction="sum")
        self.B = 2
        self.no_grids = 6
        self.lambda_noobj = 5.0
        self.lambda_coord = 0.5
        self.lambda_class = 2.0
        self.ious = []

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.compat.v1.variable_scope(scope):
            
            boxes1 = tf.stack([boxes1[:, 0,...] - boxes1[:, 2,...] / 2.0,
                               boxes1[:, 1,...] - boxes1[:, 3,...] / 2.0,
                               boxes1[:, 0,...] + boxes1[:, 2,...] / 2.0,
                               boxes1[:, 1,...] + boxes1[:, 3,...] / 2.0],axis=1)

            boxes2 = tf.stack([boxes2[:, 0,...] - boxes2[:, 2,...] / 2.0,
                               boxes2[:, 1,...] - boxes2[:, 3,...] / 2.0,
                               boxes2[:, 0,...] + boxes2[:, 2,...] / 2.0,
                               boxes2[:, 1,...] + boxes2[:, 3,...] / 2.0],axis=1)
            #boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])
            #print(tf.shape(boxes1))
            #print(boxes1[:,0:4,...])
            #print(boxes2[:,:2,...])
            # calculate the left up point & right down point
            
            lu = tf.maximum(boxes1[:, :2,...], boxes2[:, :2,...])
            rd = tf.minimum(boxes1[:, 2:,...], boxes2[:, 2:,...])
            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, 0,...] * intersection[:, 1,...]
            
            # calculate the boxs1 square and boxs2 square
            square1 = tf.multiply(tf.subtract(boxes1[:, 2,...], boxes1[:, 0,...]),tf.subtract(boxes1[:, 3,...], boxes1[:, 1,...]))
            square2 = tf.multiply(tf.subtract(boxes2[:, 2,...], boxes2[:, 0,...]),tf.subtract(boxes2[:, 3,...], boxes2[:, 1,...]))
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
            #print(inter_square)
            #print(union_square)
            #print(tf.clip_by_value(inter_square / union_square, 0.0, 1.0))
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


    def __call__(self,y_true,y_pred):
        
        y_pred = tf.reshape(y_pred,[-1,30,self.no_grids,self.no_grids])
        y_true = tf.reshape(y_true,[-1,30,self.no_grids,self.no_grids])
        

        pred_class = y_pred[:,:20,...]
        pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1)
        pred_obj = tf.reshape(tf.concat((y_pred[:,20,...],y_pred[:,25,...]),axis=1),[-1,2,self.no_grids,self.no_grids])

        true_class = y_true[:,:20,...]
        true_box = y_true[:,21:25,...]
        true_obj = tf.reshape(y_true[:,20,...],[-1,1,self.no_grids,self.no_grids])
        
        
        offset = tf.repeat([tf.range(self.no_grids,dtype=tf.float32)]
                            ,repeats=[self.no_grids],axis=0) #in order to get the centers relative to the image
        offset = tf.reshape(offset,[1,self.no_grids,self.no_grids])

        regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                       (pred_box[:, 1,...] + offset) / self.no_grids,
                                       tf.square(pred_box[:, 2,...]),
                                       tf.square(pred_box[:, 3,...])],axis=1)

        regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                       (pred_box[:, 5,...] + offset) / self.no_grids,
                                       tf.square(pred_box[:, 6,...]),
                                       tf.square(pred_box[:, 7,...])],axis=1)

        regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                       (true_box[:, 1,...] + offset) / self.no_grids,
                                       tf.square(true_box[:, 2,...]),
                                       tf.square(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
        #x1 = K.print_tensor(regression_box1[0])
        #x2 = K.print_tensor(regression_label[0])

        iou_box1 = self.calc_iou(regression_box1, regression_label) # batchsize,grid,grid
        iou_box2 = self.calc_iou(regression_box2, regression_label) # batchsize,grid,grid

        bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
        
        ## I only use the bbox with the best IOU for the loss
        ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
        ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
        best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
        best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
        best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
        
        
        #x = K.print_tensor(tf.shape(tf.expand_dims(pred_obj[:,0,...],axis=1)[0]))
        #x = K.print_tensor(tf.shape(tf.expand_dims(pred_obj[:,1,...],axis=1)[0]))
        
        object_mask = true_obj#best_iou * true_obj

        #x1 = K.print_tensor(tf.shape(object_mask))
        #x2 = K.print_tensor(best_iou[0])
        #x = K.print_tensor(object_mask[0])


        """
        BOX COORDINATE LOSS/REGRESSION LOSS
        """
        boxes_delta = object_mask * (regression_label-best_regression_box)
        
        #x1 = K.print_tensor(boxes_delta[0])
        

        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta),axis=[3,2,1]),name='coord_loss')*self.lambda_coord
        #x = K.print_tensor(coord_loss)

        """
        CLASSIFICATION LOSS
        """
        class_delta = true_obj * (true_class-pred_class)
        #x1 = K.print_tensor(pred_class[0])
        cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta),axis=[3,2,1]), name='cls_loss') * self.lambda_class
        #x = K.print_tensor(cls_loss)
        
        """
        OBJECT AND NO OBJECT LOSS
        """

        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        object_delta = object_mask * (best_iou - best_has_obj)

        #noobject_delta = noobject_mask * best_has_obj

        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[3,2,1]),  name='object_loss')
        #noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[3,2,1]), name='noobject_loss') * self.lambda_noobj
        #x4=K.print_tensor(object_loss)
        #x6 = K.print_tensor(object_mask[0]) 
        #x5=K.print_tensor(noobject_mask[0])
        
        #x8= K.print_tensor(noobject_delta[0])
        #return coord_loss+cls_loss+object_loss#+noobject_loss
        
        #x = K.print_tensor(K.mean(K.square(y_pred - y_true), axis=-1))
        return K.mean(K.square(y_pred - y_true), axis=-1)

        
"""
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
no_grids = 6 # number of grid cells per image
GT = GridTransform(B, no_grids)

bboxes_grids,labels_grids = GT.transform(data,bboxes,labels,no_objects)


one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
#print(one_head_labels[2,21,...])
#print(one_head_labels.shape)
one_head_labels = np.reshape(one_head_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))


#x1 = tf.Variable([1.,0.5,3.2,4.5],dtype=tf.float32)
#x2 = tf.Variable([0.4,0.2,1.2,1.6],dtype=tf.float32)

loss = YoloLoss()
loss(one_head_labels[1:3],one_head_labels[1:3])

"""
