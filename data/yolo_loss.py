from data.data import VOCdataset
from data.transforms import GridTransform

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

def yolo_loss(y_true, y_pred, H=224, W=224, S=7,B=2):
    #prediction: batchsize x 30 x 7 x 7 , bboxes: batchsize x (21:25+26:30) x 7 x 7 , confidences: batchsize x (20+25) x 7 x 7 , class: batchsize x (0:20) x 7 x 7 
    #labels: batchsize x 25 x 7 x 7 , response_mask: batchsize x (20) x 7 x 7 , bbox: batchsize x (21:25) x 7 x 7 , class: batchsize x (:20) x 7 x 7 
    y_pred = tf.reshape(y_pred,[-1,(len(classes)+B*5),S,S])
    y_true = tf.reshape(y_true,[-1,(len(classes)+B*5),S,S])
    
    pred_class = y_pred[:,:len(classes),...] # batch x 20 x 7 x 7
    pred_bboxes = tf.concat((tf.expand_dims(y_pred[:,len(classes)+1:len(classes)+5,...],axis=1),tf.expand_dims(y_pred[:,len(classes)+6:len(classes)+10,...],axis=1)),axis=1) # batch x 2 x 4 x 7 x 7
    pred_confidence = tf.concat((tf.expand_dims(y_pred[:,len(classes),...],axis=1),tf.expand_dims(y_pred[:,len(classes)+5,...],axis=1)),axis=1) # batch x 2 x 7 x 7

    label_class = y_true[:,:len(classes),...] #batch x 20 x 7 x 7
    label_bboxes = tf.reshape(tf.tile(y_true[:,len(classes)+1:len(classes)+5,...],multiples=[1,2,1,1]),[-1,2,4,S,S]) #batch x 2 x 4 x 7 x 7
    label_mask = y_true[:,len(classes):len(classes)+1,...] #batch x 1  x 7 x 7

    #K.print_tensor(tf.shape(pred_class))
    #K.print_tensor(tf.shape(pred_bboxes))
    #K.print_tensor(tf.shape(pred_confidence))
    
    #K.print_tensor(tf.shape(label_class))
    #K.print_tensor(tf.shape(label_bboxes))
    #K.print_tensor(tf.shape(label_mask))

    """
    Bring the coordinates from YOLO format(position relative to cell), to position relative to the image height and width
    """
    # cell width and height
    cell_h = W/ S
    cell_w = H/ S
    
    """
    Offsets for columns and rows
    """
    temp = tf.constant([[0., 1., 2., 3., 4., 5., 6.]])
    temp = tf.tile(temp, multiples=[S, 1])

    col = tf.tile(temp[tf.newaxis, tf.newaxis, tf.newaxis, :, :],multiples=[1,2,1,1,1]) #1 x 1 x 1 x 7 x 7
    row = tf.transpose(col, perm=[0, 1, 2, 4, 3])

    pred_bboxes_original = tf.concat([(pred_bboxes[:, :, 0:1, :, :] + col) * cell_w,
                                      (pred_bboxes[:, :, 1:2, :, :] + row) * cell_h,
                                       pred_bboxes[:, :, 2:3, :, :] * W,
                                       pred_bboxes[:, :, 3:4, :, :] * H], axis=2)

    label_bboxes_original = tf.concat([(label_bboxes[:, :, 0:1, :, :] + col) * cell_w,
                                      (label_bboxes[:, :, 1:2, :, :] + row) * cell_h,
                                      label_bboxes[:, :, 2:3, :, :] * W,
                                      label_bboxes[:, :, 3:4, :, :] * H], axis=2)

    """
    The IOU is calculated for each bounding box prediction and the ground truth. In total there are 98 IoUs.
    Then the max IOU between the 2 bounding box predictions per cell is calculated in order to use only the predictions that have the highest IoU for calculating loss.
    """

    iou = cal_iou(pred_bboxes_original, label_bboxes_original) # Batch x 2 x 7 x 7

    max_iou = tf.reduce_max(iou, axis=1, keepdims=True)#output: batchsize x 1 x 7 x7

    pred_bboxes_sqrt = tf.concat([pred_bboxes[:, :, 0:1, :, :],
                                      pred_bboxes[:, :, 1:2, :, :],
                                       tf.sqrt(pred_bboxes[:, :, 2:3, :, :]),
                                       tf.sqrt(pred_bboxes[:, :, 3:4, :, :])], axis=2)

    label_bboxes_sqrt = tf.concat([label_bboxes[:, :, 0:1, :, :],
                                      label_bboxes[:, :, 1:2, :, :],
                                      tf.sqrt(label_bboxes[:, :, 2:3, :, :]),
                                      tf.sqrt(label_bboxes[:, :, 3:4, :, :])], axis=2)

    # In the places where there should be an object, we keep only the cells with the greatest IoU out of the 2 bounding box predictions for calculating the loss.
    # Where there should be an object and the IoU is the max one out of the 2 bboxes: value=1, otherwise =0
    mask_obj = tf.cast(tf.greater_equal(iou, max_iou), dtype=tf.float32) * label_mask #batch x 2 x 7 x 7

    loss_bboxes = tf.reduce_mean(tf.reduce_sum(tf.square(pred_bboxes_sqrt - label_bboxes_sqrt) * mask_obj[:, :, tf.newaxis, :, :], axis=[1, 2, 3, 4]))

    loss_confidence_obj = tf.reduce_mean(tf.reduce_sum(tf.square(iou*pred_confidence - 1) * mask_obj, axis=[1, 2, 3])) #objectness = iou*pred_confidence

    loss_confidence_noobj = tf.reduce_mean(tf.reduce_sum(tf.square(iou*pred_confidence) * (1 - label_mask), axis=[1, 2, 3])) #objectness = iou*pred_confidence
    
    loss_class = tf.reduce_mean(tf.reduce_sum(tf.square(pred_class - label_class) * label_mask, axis=[1, 2, 3]))
    
    loss = 5 * loss_bboxes + loss_confidence_obj + 0.5 * loss_confidence_noobj + loss_class
    
    return loss#, loss_bboxes, loss_confidence_obj, loss_confidence_noobj, loss_class

def cal_iou(bboxes1, bboxes2):
    #bboxes: [batchsize, 2, 4, 7, 7] with [center_x, center_y, h, w]
    cx, cx_ = bboxes1[:, :, 0, :, :], bboxes2[:, :, 0, :, :] # batch,2,7,7
    cy, cy_ = bboxes1[:, :, 1, :, :], bboxes2[:, :, 1, :, :] # batch,2,7,7
    w, w_ = bboxes1[:, :, 2, :, :], bboxes2[:, :, 2, :, :] # batch,2,7,7
    h, h_ = bboxes1[:, :, 3, :, :], bboxes2[:, :, 3, :, :] # batch,2,7,7
    x1, x1_ = cx - w / 2, cx_ - w_ / 2
    x2, x2_ = cx + w / 2, cx_ + w_ / 2
    y1, y1_ = cy - h / 2, cy_ - h_ / 2
    y2, y2_ = cy + h / 2, cy_ + h_ / 2
    x_inter1 = tf.maximum(x1, x1_)
    x_inter2 = tf.minimum(x2, x2_)
    y_inter1 = tf.maximum(y1, y1_)
    y_inter2 = tf.minimum(y2, y2_)
    h_inter = tf.maximum(0., y_inter2 - y_inter1)
    w_inter = tf.maximum(0., x_inter2 - x_inter1)
    area_inter = h_inter * w_inter
    area_union = tf.maximum(h * w + h_ * w_ - area_inter,1e-10)
    iou = tf.clip_by_value(area_inter / area_union, 0.0, 1.0)
    return iou
    
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
print(imagePaths[2])
#default number of grids: no_grids =11 and default number of box predictions: B=2

B = 2 # number of bbox predictions per cell
no_grids = 7 # number of grid cells per image
GT = GridTransform(B, no_grids)

bboxes_grids,labels_grids = GT.transform(bboxes,labels,no_objects)


one_head_labels = np.concatenate((labels_grids,bboxes_grids),axis=1)
#print(one_head_labels[2,21,...])
#print(one_head_labels.shape)
#print(one_head_labels[1,21:25,...])

one_head_labels = np.reshape(one_head_labels,(-1,30*7*7))


#loss = YoloLoss()

print(yolo_loss(one_head_labels[2],one_head_labels[2]+0.05))
"""
