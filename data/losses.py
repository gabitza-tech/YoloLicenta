from cmath import log
import numpy as np
from tensorflow import keras
#from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error as mse
from data.data import VOCdataset
from data.transforms import GridTransform
from tensorflow.keras import backend as K
import tensorflow as tf
import collections

tf.compat.v1.enable_eager_execution()

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class YoloLoss:
        
    def __init__(self):
        #self.mse = MeanSquaredError(reduction="sum")
        self.B = 2
        self.no_grids = 7
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0
        self.lambda_class = 1.0
        self.ious = []

    def tf_count(self,t, val):
        elements_equal_to_value = tf.equal(t, val)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        return count


    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        """
        
        with tf.compat.v1.variable_scope(scope):
            # transform boxes from (cx,cy,w,h) format to (x1,y1,x2,y2) format
            boxes1 = tf.stack([boxes1[:, 0,...] - boxes1[:, 2,...] / 2.0,
                               boxes1[:, 1,...] - boxes1[:, 3,...] / 2.0,
                               boxes1[:, 0,...] + boxes1[:, 2,...] / 2.0,
                               boxes1[:, 1,...] + boxes1[:, 3,...] / 2.0],axis=1)

            boxes2 = tf.stack([boxes2[:, 0,...] - boxes2[:, 2,...] / 2.0,
                               boxes2[:, 1,...] - boxes2[:, 3,...] / 2.0,
                               boxes2[:, 0,...] + boxes2[:, 2,...] / 2.0,
                               boxes2[:, 1,...] + boxes2[:, 3,...] / 2.0],axis=1)
            
            
            # debug bun pana aici
            # calculate the left up point & right down point
            
            lu = tf.maximum(boxes1[:, :2,...], boxes2[:, :2,...])
            rd = tf.minimum(boxes1[:, 2:,...], boxes2[:, 2:,...])
            
            # debug bun 
            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, 0,...] * intersection[:, 1,...]
            #debug bun

            # calculate the boxs1 square and boxs2 square
            square1 = tf.multiply(tf.subtract(boxes1[:, 2,...], boxes1[:, 0,...]),tf.subtract(boxes1[:, 3,...], boxes1[:, 1,...]))
            square2 = tf.multiply(tf.subtract(boxes2[:, 2,...], boxes2[:, 0,...]),tf.subtract(boxes2[:, 3,...], boxes2[:, 1,...]))
            #debug bun

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
            
            # debug bun
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def __call__(self,y_true,y_pred):
        
        y_pred = tf.reshape(y_pred,[-1,30,self.no_grids,self.no_grids])
        y_true = tf.reshape(y_true,[-1,30,self.no_grids,self.no_grids])
        
        #debug uit functioneaza reshape-ul cum trebuie atat ptr groundtruth cat si pentru prediction uri
        pred_class = y_pred[:,:20,...] #batch, 20, 6, 6
        pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1) # batch, 8 ,6 ,6
        pred_obj = tf.concat((tf.expand_dims(y_pred[:,20,...],axis=1),tf.expand_dims(y_pred[:,25,...],axis=1)),axis=1)

        true_class = y_true[:,:20,...] #batch, 20,6,6
        true_box = y_true[:,21:25,...] #batch, 4, 6, 6
        true_obj = tf.reshape(y_true[:,20,...],[-1,1,self.no_grids,self.no_grids]) # batch, 1, 6, 6
        
        no_obj_batch = self.tf_count(true_obj,1)

        #debug uit functioneaza partajarea pe functionalitati
        
        offset = tf.repeat([tf.range(self.no_grids,dtype=tf.float32)]
                            ,repeats=[self.no_grids],axis=0) #in order to get the centers relative to the image
        offset = tf.reshape(offset,[1,self.no_grids,self.no_grids])

        #offsetul e bun

        box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                       (pred_box[:, 1,...] + offset) / self.no_grids,
                                       pred_box[:, 2,...],
                                       pred_box[:, 3,...]],axis=1)

        box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                       (pred_box[:, 5,...] + offset) / self.no_grids,
                                       pred_box[:, 6,...],
                                       pred_box[:, 7,...]],axis=1)

        reg_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                       (true_box[:, 1,...] + offset) / self.no_grids,
                                       true_box[:, 2,...],
                                       true_box[:, 3,...]],axis=1)  # batchsize, 4,grid,grid
        
        iou_box1 = self.calc_iou(box1, reg_label) # batchsize,grid,grid
        iou_box2 = self.calc_iou(box2, reg_label) # batchsize,grid,grid
        
        """
#### INCERCAT FARA OFFSET BBOX LOSS
        regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                       (pred_box[:, 1,...] + offset) / self.no_grids,
                                       tf.sqrt(pred_box[:, 2,...]),
                                       tf.sqrt(pred_box[:, 3,...])],axis=1)

        regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                       (pred_box[:, 5,...] + offset) / self.no_grids,
                                       tf.sqrt(pred_box[:, 6,...]),
                                       tf.sqrt(pred_box[:, 7,...])],axis=1)

        regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                       (true_box[:, 1,...] + offset) / self.no_grids,
                                       tf.sqrt(true_box[:, 2,...]),
                                       tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
"""
        regression_box1 = tf.stack([pred_box[:, 0, ...] ,
                                              pred_box[:, 1,...],
                                              tf.sqrt(pred_box[:, 2,...]),
                                              tf.sqrt(pred_box[:, 3,...])],axis=1)

        regression_box2 = tf.stack([pred_box[:, 4, ...],
                                        pred_box[:, 5,...] ,
                                        tf.sqrt(pred_box[:, 6,...]),
                                        tf.sqrt(pred_box[:, 7,...])],axis=1)

        regression_label = tf.stack([true_box[:, 0, ...] ,
                                        true_box[:, 1,...],
                                        tf.sqrt(true_box[:, 2,...]),
                                        tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid

        bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
        
        # debug bun pana aici
        ## I only use the bbox with the best IOU for the loss
        ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
        ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
        best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
        best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
        best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
        #debug bun pana aici
        
        

        """
        BOX COORDINATE LOSS/REGRESSION LOSS
        """
        boxes_delta = true_obj * (regression_label-best_regression_box)

        #batch,4,6,6

        coord_loss = tf.reduce_sum(tf.square(boxes_delta),axis=[3,2,1,0])*self.lambda_coord/tf.cast(tf.shape(y_pred)[0],tf.float32)
        #print("coord loss:{}".format(coord_loss))
        #x = K.print_tensor(coord_loss)
        
        #debug bun

        """
        CLASSIFICATION LOSS
        """
        #loss = label * (-1) * log(pred) - (1 - label) * (-1) * log(1 - pred)
        class_delta = true_obj * (true_class-pred_class)
        #class_delta = (-1)*(true_obj*true_class)*tf.math.log(true_obj*pred_class+tf.keras.backend.epsilon())-(tf.ones_like(true_class,dtype=tf.float32)-true_obj*true_class)*tf.math.log(tf.ones_like(pred_class,dtype=tf.float32)-pred_class*true_obj+tf.keras.backend.epsilon())

        #debug bun 
        
        cls_loss = tf.reduce_sum(class_delta,axis=[3,2,1,0])*self.lambda_class/tf.cast(tf.shape(y_pred)[0],tf.float32)
        
        #print("class los:{}".format(cls_loss))

        
        #debug bun
        """
        OBJECT AND NO OBJECT LOSS
        """

        object_mask = best_iou * true_obj#true_obj#
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
                
        object_delta = true_obj*(object_mask - best_has_obj)#object_mask * (true_obj - best_has_obj)#
        noobject_delta = noobject_mask * pred_obj
        
        
        object_loss = tf.reduce_sum(tf.square(object_delta), axis=[3,2,1,0]) * self.lambda_obj/tf.cast(tf.shape(y_pred)[0],tf.float32)
        noobject_loss = tf.reduce_sum(tf.square(noobject_delta), axis=[3,2,1,0]) * self.lambda_noobj/tf.cast(tf.shape(y_pred)[0],tf.float32)
        
        #print("object loss:{}".format(object_loss))
        #print("no object loss:{}".format(noobject_loss))

        #debug bun
        
        loss = (coord_loss+cls_loss+object_loss+noobject_loss)

        return loss 
        #x = K.print_tensor(K.mean(K.square(y_pred - y_true), axis=-1))
        #return K.mean(K.square(y_pred - y_true), axis=-1)

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count
def calc_iou( boxes1, boxes2, scope='iou'):
    """calculate ious
    """
    
    with tf.compat.v1.variable_scope(scope):
        # transform boxes from (cx,cy,w,h) format to (x1,y1,x2,y2) format
        boxes1 = tf.stack([boxes1[:, 0,...] - boxes1[:, 2,...] / 2.0,
                            boxes1[:, 1,...] - boxes1[:, 3,...] / 2.0,
                            boxes1[:, 0,...] + boxes1[:, 2,...] / 2.0,
                            boxes1[:, 1,...] + boxes1[:, 3,...] / 2.0],axis=1)

        boxes2 = tf.stack([boxes2[:, 0,...] - boxes2[:, 2,...] / 2.0,
                            boxes2[:, 1,...] - boxes2[:, 3,...] / 2.0,
                            boxes2[:, 0,...] + boxes2[:, 2,...] / 2.0,
                            boxes2[:, 1,...] + boxes2[:, 3,...] / 2.0],axis=1)
        
        
        # debug bun pana aici
        # calculate the left up point & right down point
        
        lu = tf.maximum(boxes1[:, :2,...], boxes2[:, :2,...])
        rd = tf.minimum(boxes1[:, 2:,...], boxes2[:, 2:,...])
        
        # debug bun 
        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, 0,...] * intersection[:, 1,...]
        #debug bun

        # calculate the boxs1 square and boxs2 square
        square1 = tf.multiply(tf.subtract(boxes1[:, 2,...], boxes1[:, 0,...]),tf.subtract(boxes1[:, 3,...], boxes1[:, 1,...]))
        square2 = tf.multiply(tf.subtract(boxes2[:, 2,...], boxes2[:, 0,...]),tf.subtract(boxes2[:, 3,...], boxes2[:, 1,...]))
        #debug bun

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        
        # debug bun
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    
def class_loss(y_true,y_pred):      
    no_grids=7
    y_pred = tf.reshape(y_pred,[-1,30,no_grids,no_grids])
    y_true = tf.reshape(y_true,[-1,30,no_grids,no_grids])
    
    #debug uit functioneaza reshape-ul cum trebuie atat ptr groundtruth cat si pentru prediction uri
    #debug uit functioneaza reshape-ul cum trebuie atat ptr groundtruth cat si pentru prediction uri
    pred_class = y_pred[:,:20,...] #batch, 20, 6, 6
    pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1) # batch, 8 ,6 ,6
    pred_obj = tf.reshape(tf.concat((y_pred[:,20,...],y_pred[:,25,...]),axis=1),[-1,2,no_grids,no_grids]) # batch, 2, 6, 6

    true_class = y_true[:,:20,...] #batch, 20,6,6
    true_box = y_true[:,21:25,...] #batch, 4, 6, 6
    true_obj = tf.reshape(y_true[:,20,...],[-1,1,no_grids,no_grids]) # batch, 1, 6, 6
    
    no_obj_batch = tf_count(true_obj,1)

    #debug uit functioneaza partajarea pe functionalitati
    
    offset = tf.repeat([tf.range(no_grids,dtype=tf.float32)]
                        ,repeats=[no_grids],axis=0) #in order to get the centers relative to the image
    offset = tf.reshape(offset,[1,no_grids,no_grids])
    #offsetul e bun

    box1 = tf.stack([(pred_box[:, 0, ...] + offset) / no_grids,
                                    (pred_box[:, 1,...] + offset) / no_grids,
                                    pred_box[:, 2,...],
                                    pred_box[:, 3,...]],axis=1)

    box2 = tf.stack([(pred_box[:, 4, ...] + offset) / no_grids,
                                    (pred_box[:, 5,...] + offset) / no_grids,
                                    pred_box[:, 6,...],
                                    pred_box[:, 7,...]],axis=1)

    reg_label = tf.stack([(true_box[:, 0, ...] + offset) / no_grids,
                                    (true_box[:, 1,...] + offset) / no_grids,
                                    true_box[:, 2,...],
                                    true_box[:, 3,...]],axis=1)  # batchsize, 4,grid,grid
    
    iou_box1 = calc_iou(box1, reg_label) # batchsize,grid,grid
    iou_box2 = calc_iou(box2, reg_label) # batchsize,grid,grid
    
    """
#### INCERCAT FARA OFFSET BBOX LOSS
    regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                    (pred_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 2,...]),
                                    tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                    (pred_box[:, 5,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                    (true_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
"""
    regression_box1 = tf.stack([pred_box[:, 0, ...] ,
                                            pred_box[:, 1,...],
                                            tf.sqrt(pred_box[:, 2,...]),
                                            tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([pred_box[:, 4, ...],
                                    pred_box[:, 5,...] ,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([true_box[:, 0, ...] ,
                                    true_box[:, 1,...],
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid

    bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
    
    # debug bun pana aici
    ## I only use the bbox with the best IOU for the loss
    ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
    ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
    best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
    best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
    best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
    #debug bun pana aici
    
    """
    CLASSIFICATION LOSS
    """
    #loss = label * (-1) * log(pred) + (1 - label) * (-1) * log(1 - pred)
    class_delta = (-1)*(true_obj*true_class)*tf.math.log(true_obj*pred_class+tf.keras.backend.epsilon())-(tf.ones_like(true_class,dtype=tf.float32)-true_obj*true_class)*tf.math.log(tf.ones_like(pred_class,dtype=tf.float32)-pred_class*true_obj+tf.keras.backend.epsilon())

        #debug bun 
        
    cls_loss = tf.reduce_sum(class_delta,axis=[3,2,1,0])/tf.cast(no_obj_batch,tf.float32)
        
    
    return cls_loss

def box_loss(y_true,y_pred):      
    no_grids=7
    y_pred = tf.reshape(y_pred,[-1,30,no_grids,no_grids])
    y_true = tf.reshape(y_true,[-1,30,no_grids,no_grids])
    

    pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1) # batch, 8 ,6 ,6
    pred_obj = tf.reshape(tf.concat((y_pred[:,20,...],y_pred[:,25,...]),axis=1),[-1,2,no_grids,no_grids]) # batch, 2, 6, 6

    
    true_box = y_true[:,21:25,...] #batch, 4, 6, 6
    true_obj = tf.reshape(y_true[:,20,...],[-1,1,no_grids,no_grids]) # batch, 1, 6, 6
    
    no_obj_batch = tf_count(true_obj,1)
    #debug uit functioneaza partajarea pe functionalitati
    
    offset = tf.repeat([tf.range(no_grids,dtype=tf.float32)]
                        ,repeats=[no_grids],axis=0) #in order to get the centers relative to the image
    offset = tf.reshape(offset,[1,no_grids,no_grids])
    #offsetul e bun

    box1 = tf.stack([(pred_box[:, 0, ...] + offset) / no_grids,
                                    (pred_box[:, 1,...] + offset) / no_grids,
                                    pred_box[:, 2,...],
                                    pred_box[:, 3,...]],axis=1)

    box2 = tf.stack([(pred_box[:, 4, ...] + offset) / no_grids,
                                    (pred_box[:, 5,...] + offset) / no_grids,
                                    pred_box[:, 6,...],
                                    pred_box[:, 7,...]],axis=1)

    reg_label = tf.stack([(true_box[:, 0, ...] + offset) / no_grids,
                                    (true_box[:, 1,...] + offset) / no_grids,
                                    true_box[:, 2,...],
                                    true_box[:, 3,...]],axis=1)  # batchsize, 4,grid,grid
    
    iou_box1 = calc_iou(box1, reg_label) # batchsize,grid,grid
    iou_box2 = calc_iou(box2, reg_label) # batchsize,grid,grid
    
    """
#### INCERCAT FARA OFFSET BBOX LOSS
    regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                    (pred_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 2,...]),
                                    tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                    (pred_box[:, 5,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                    (true_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
"""
    regression_box1 = tf.stack([pred_box[:, 0, ...] ,
                                            pred_box[:, 1,...],
                                            tf.sqrt(pred_box[:, 2,...]),
                                            tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([pred_box[:, 4, ...],
                                    pred_box[:, 5,...] ,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([true_box[:, 0, ...] ,
                                    true_box[:, 1,...],
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid

    bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
    
    # debug bun pana aici
    ## I only use the bbox with the best IOU for the loss
    ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
    ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
    best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
    best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
    best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
    #debug bun pana aici
    
    """
    BOX COORDINATE LOSS/REGRESSION LOSS
    """
    boxes_delta = true_obj * (regression_label-best_regression_box)

    #batch,4,6,6

    coord_loss = tf.reduce_sum(tf.square(boxes_delta),axis=[3,2,1,0])*5.0/tf.cast(no_obj_batch,tf.float32)
    #x = K.print_tensor(coord_loss)
    
    #debug bun
    
    return coord_loss

def obj_loss(y_true,y_pred):      
    no_grids=7
    y_pred = tf.reshape(y_pred,[-1,30,no_grids,no_grids])
    y_true = tf.reshape(y_true,[-1,30,no_grids,no_grids])
    
    pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1) # batch, 8 ,6 ,6
    pred_obj = tf.reshape(tf.concat((y_pred[:,20,...],y_pred[:,25,...]),axis=1),[-1,2,no_grids,no_grids]) # batch, 2, 6, 6

    true_box = y_true[:,21:25,...] #batch, 4, 6, 6
    true_obj = tf.reshape(y_true[:,20,...],[-1,1,no_grids,no_grids]) # batch, 1, 6, 6
    
    no_obj_batch = tf_count(true_obj,1)
    #debug uit functioneaza partajarea pe functionalitati
    
    offset = tf.repeat([tf.range(no_grids,dtype=tf.float32)]
                        ,repeats=[no_grids],axis=0) #in order to get the centers relative to the image
    offset = tf.reshape(offset,[1,no_grids,no_grids])
    #offsetul e bun

    box1 = tf.stack([(pred_box[:, 0, ...] + offset) / no_grids,
                                    (pred_box[:, 1,...] + offset) / no_grids,
                                    pred_box[:, 2,...],
                                    pred_box[:, 3,...]],axis=1)

    box2 = tf.stack([(pred_box[:, 4, ...] + offset) / no_grids,
                                    (pred_box[:, 5,...] + offset) / no_grids,
                                    pred_box[:, 6,...],
                                    pred_box[:, 7,...]],axis=1)

    reg_label = tf.stack([(true_box[:, 0, ...] + offset) / no_grids,
                                    (true_box[:, 1,...] + offset) / no_grids,
                                    true_box[:, 2,...],
                                    true_box[:, 3,...]],axis=1)  # batchsize, 4,grid,grid
    
    iou_box1 = calc_iou(box1, reg_label) # batchsize,grid,grid
    iou_box2 = calc_iou(box2, reg_label) # batchsize,grid,grid
    
    """
#### INCERCAT FARA OFFSET BBOX LOSS
    regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                    (pred_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 2,...]),
                                    tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                    (pred_box[:, 5,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                    (true_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
"""
    regression_box1 = tf.stack([pred_box[:, 0, ...] ,
                                            pred_box[:, 1,...],
                                            tf.sqrt(pred_box[:, 2,...]),
                                            tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([pred_box[:, 4, ...],
                                    pred_box[:, 5,...] ,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([true_box[:, 0, ...] ,
                                    true_box[:, 1,...],
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid

    bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
    
    # debug bun pana aici
    ## I only use the bbox with the best IOU for the loss
    ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
    ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
    best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
    best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
    best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
    #debug bun pana aici
    #debug bun
    """
    OBJECT AND NO OBJECT LOSS
    """

    object_mask = best_iou * true_obj#true_obj#

            
    object_delta = true_obj*(object_mask - best_has_obj)#object_mask * (true_obj - best_has_obj)#

    
    object_loss = tf.reduce_sum(tf.square(object_delta), axis=[3,2,1,0])/tf.cast(no_obj_batch,tf.float32)
    
    return object_loss

def noobj_loss(y_true,y_pred):      
    no_grids=7
    y_pred = tf.reshape(y_pred,[-1,30,no_grids,no_grids])
    y_true = tf.reshape(y_true,[-1,30,no_grids,no_grids])
       

    pred_box = tf.concat((y_pred[:,21:25,...],y_pred[:,26:30,...]),axis=1) # batch, 8 ,6 ,6
    pred_obj = tf.reshape(tf.concat((y_pred[:,20,...],y_pred[:,25,...]),axis=1),[-1,2,no_grids,no_grids]) # batch, 2, 6, 6

    
    true_box = y_true[:,21:25,...] #batch, 4, 6, 6
    true_obj = tf.reshape(y_true[:,20,...],[-1,1,no_grids,no_grids]) # batch, 1, 6, 6
    
    #debug uit functioneaza partajarea pe functionalitati
    
    offset = tf.repeat([tf.range(no_grids,dtype=tf.float32)]
                        ,repeats=[no_grids],axis=0) #in order to get the centers relative to the image
    offset = tf.reshape(offset,[1,no_grids,no_grids])
    #offsetul e bun

    box1 = tf.stack([(pred_box[:, 0, ...] + offset) / no_grids,
                                    (pred_box[:, 1,...] + offset) / no_grids,
                                    pred_box[:, 2,...],
                                    pred_box[:, 3,...]],axis=1)

    box2 = tf.stack([(pred_box[:, 4, ...] + offset) / no_grids,
                                    (pred_box[:, 5,...] + offset) / no_grids,
                                    pred_box[:, 6,...],
                                    pred_box[:, 7,...]],axis=1)

    reg_label = tf.stack([(true_box[:, 0, ...] + offset) / no_grids,
                                    (true_box[:, 1,...] + offset) / no_grids,
                                    true_box[:, 2,...],
                                    true_box[:, 3,...]],axis=1)  # batchsize, 4,grid,grid
    
    iou_box1 = calc_iou(box1, reg_label) # batchsize,grid,grid
    iou_box2 = calc_iou(box2, reg_label) # batchsize,grid,grid
    
    """
#### INCERCAT FARA OFFSET BBOX LOSS
    regression_box1 = tf.stack([(pred_box[:, 0, ...] + offset) / self.no_grids,
                                    (pred_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 2,...]),
                                    tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([(pred_box[:, 4, ...] + offset) / self.no_grids,
                                    (pred_box[:, 5,...] + offset) / self.no_grids,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    regression_label = tf.stack([(true_box[:, 0, ...] + offset) / self.no_grids,
                                    (true_box[:, 1,...] + offset) / self.no_grids,
                                    tf.sqrt(true_box[:, 2,...]),
                                    tf.sqrt(true_box[:, 3,...])],axis=1)  # batchsize, 4,grid,grid
"""
    regression_box1 = tf.stack([pred_box[:, 0, ...] ,
                                            pred_box[:, 1,...],
                                            tf.sqrt(pred_box[:, 2,...]),
                                            tf.sqrt(pred_box[:, 3,...])],axis=1)

    regression_box2 = tf.stack([pred_box[:, 4, ...],
                                    pred_box[:, 5,...] ,
                                    tf.sqrt(pred_box[:, 6,...]),
                                    tf.sqrt(pred_box[:, 7,...])],axis=1)

    bestbox = tf.math.argmax((iou_box1,iou_box2),axis=0) #batchsize, grid, grid
    
    # debug bun pana aici
    ## I only use the bbox with the best IOU for the loss
    ## (1-bestbox)*iou_box1+bestbox*iou_box2 == best_iou   batch_size,grid,grid
    ## (1-bestbox)*reg_box1 + bestbox*iou_box2 == best_reg_box   batch_size,4,grid,grid
    best_iou = tf.expand_dims(tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(bestbox,dtype=tf.int32)),dtype=tf.float32),iou_box1),tf.cast(tf.multiply(tf.cast(bestbox,dtype=tf.float32),iou_box2),dtype=tf.float32)),axis=1)
    best_regression_box = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),regression_box1),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),regression_box2),dtype=tf.float32))
    best_has_obj = tf.add(tf.multiply(tf.cast(tf.subtract(1,tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.int32)),dtype=tf.float32),tf.expand_dims(pred_obj[:,0,...],axis=1)),tf.cast(tf.multiply(tf.cast(tf.expand_dims(bestbox,axis=1),dtype=tf.float32),tf.expand_dims(pred_obj[:,1,...],axis=1)),dtype=tf.float32))
    #debug bun pana aici
    #debug bun
    """
    OBJECT AND NO OBJECT LOSS
    """

    object_mask = best_iou * true_obj#true_obj#
    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
            

    noobject_delta = noobject_mask * pred_obj
    
    noobject_loss = tf.reduce_sum(tf.square(noobject_delta), axis=[3,2,1,0]) /tf.cast(tf.shape(y_pred)[0],tf.float32)*0.5
    return noobject_loss

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

bboxes_grids,labels_grids = GT.transform(bboxes,labels,no_objects)


one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
#print(one_head_labels[2,21,...])
#print(one_head_labels.shape)
#print(one_head_labels[1,21:25,...])

#one_head_labels = np.reshape(one_head_labels,(-1,(len(classes)+B*5)*no_grids*no_grids))


#loss = YoloLoss()

y_true = [[0, 1, 0,1,0,1], [0, 0, 1,0,0,1]]
y_pred = [[0.05, 0.95, 0.3,0,0.95,0], [0.1, 0.8, 0.1,1,0,1]]
loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
print(loss)
"""

