from data.data import VOCdataset
import argparse
import numpy as np
import cv2
import tensorflow as tf

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

class GridTransform:
    def __init__(self,B=2,no_grids=7):
        self.bboxes_tr = []
        self.labels_tr = []
        self.no_grids = no_grids
        self.B = B
        self.w_cell = 1/no_grids
        self.h_cell = 1/no_grids

    def cell_pos_relative(self,bboxes):
        cell_pos_per_image = []
        cell_centers_per_image = []

        for bbox_image in bboxes:
            cell_pos = []
            cell_centers = []
            for bbox in bbox_image:
                pos_cell_column = int(bbox[0]/self.w_cell) ## coloana, x
                pos_cell_row = int(bbox[1]/self.h_cell) ## linie, y
                relative_cx = (bbox[0]-pos_cell_column*self.w_cell)/self.w_cell
                relative_cy = (bbox[1]-pos_cell_row*self.h_cell)/self.h_cell

                cell_pos.append([pos_cell_column,pos_cell_row])
                cell_centers.append([relative_cx, relative_cy, bbox[2], bbox[3]])

            cell_pos_per_image.append(cell_pos)
            cell_centers_per_image.append(cell_centers)

        return cell_pos_per_image, cell_centers_per_image
    
    def transform(self,bboxes,labels,no_objects):
        cell_pos_image, cell_centers_image = self.cell_pos_relative(bboxes)
        
        for i in range(len(no_objects)):
            bboxes_grid = np.zeros((self.B*5,self.no_grids,self.no_grids))
            label_grid = np.zeros((len(classes),self.no_grids,self.no_grids))
            for j in range(no_objects[i]):
                
                # folosesc [valoare,row,column]

                #BBOX 1
                bboxes_grid[1][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][0]#cx
                bboxes_grid[2][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][1]#cy
                bboxes_grid[3][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][2]#w
                bboxes_grid[4][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][3]#h
                
                bboxes_grid[0][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1 ## has_obj matrix
                
                label_grid[labels[i][j]][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1 # label

            self.bboxes_tr.append(bboxes_grid)
            self.labels_tr.append(label_grid)

        self.bboxes_tr = np.array(self.bboxes_tr,dtype='float32')
        self.labels_tr = np.array(self.labels_tr,dtype='float32')
        return self.bboxes_tr,self.labels_tr

    def iou_value(self,box1, box2, box1_pos, box2_pos):
        '''
        calculate the IOU of two given boxes
        '''
        (c1x_cell,c1y_cell,w1,h1) = box1
        (c2x_cell,c2y_cell,w2,h2) = box2

        """
        first transform the centers relative to a cell to centers relative to the image
        """
        
        row1 = box1_pos // self.no_grids
        col1 = box1_pos % self.no_grids
        
        row2 = box2_pos // self.no_grids
        col2 = box2_pos % self.no_grids

        c1x_img = col1/self.no_grids + c1x_cell/self.no_grids
        c1y_img = row1/self.no_grids + c1y_cell/self.no_grids
        
        c2x_img = col2/self.no_grids + c2x_cell/self.no_grids
        c2y_img = row2/self.no_grids + c2y_cell/self.no_grids
        """
        now calculate the upper left corners and the lower right corners
        """
        (x11, y11 , x12, y12) = (c1x_img-w1/2,c1y_img-h1/2, c1x_img+w1/2,c1y_img+h1/2)
        (x21, y21 , x22, y22) = (c2x_img-w2/2,c2y_img-h2/2, c2x_img+w2/2,c2y_img+h2/2)

        """
        calculate the intersection points,width and height
        """
        x1 = max(x11, x21)
        x2 = min(x12, x22)
        w = max(0, (x2-x1))

        y1 = max(y11, y21)
        y2 = min(y12, y22)
        h = max(0, (y2-y1))

        area_intersection = w*h

        area_combined = abs((x12-x11)*(y12-y11) + (x22-x21)*(y22-y21) - area_intersection + 1e-3)
        return area_intersection/area_combined


    def nonmax_suppression(self,bboxes,labels,initial_positions, iou_cutoff):
        '''
        Suppress any overlapping boxes with IOU greater than 'iou_cutoff', keeping only
        the one with highest confidence scores
        # Arguments
        bboxes: array of ((x1,y1), (x2,y2)), c) where c is the confidence score
        iou_cutoff: any IOU greater than this is considered for suppression
        '''
        
        suppress_list = []
        nms_list = []
        positions = []
        """
        compare every cell prediction with the other cell predictions
        note: may not be the most efficient
        """
        
        for i in range(bboxes.shape[0]):

            box1 = bboxes[i]
            
            label1 = labels[i]
            label_pos1 = np.argmax(label1[:20],axis=0)
            
            for j in range(i+1, bboxes.shape[0]):
            
                box2 = bboxes[j]

                label2 = labels[j]
                label_pos2 = np.argmax(label2[:20],axis=0)
                
                iou = self.iou_value(box1[1:5], box2[1:5],initial_positions[i],initial_positions[j])
                #print(initial_positions[i], " & ", initial_positions[j], "IOU: ", iou)
                if label_pos1 == label_pos2:
                    if iou >= iou_cutoff:
                        if box1[0] > box2[0]:
                            suppress_list.append(j)
                        else:
                            suppress_list.append(i)
                            continue
        #print('suppress_list: ', suppress_list)

        for i in range(bboxes.shape[0]):
            if i in suppress_list:
                continue
            else:
                nms_list.append(bboxes[i])
                positions.append(initial_positions[i])
        nms_list = np.asarray(nms_list)
        return nms_list, positions

    def transform_with_nms(self,bboxes,labels,image,conf_thresh = 0.8,nms_iou_cutoff = 0.05):
        (h_img, w_img) = image.shape[:2]
        M = h_img//self.no_grids
        N = w_img//self.no_grids

        for y in range(0,h_img,M):
            for x in range(0, w_img, N):
                y1 = y + M
                x1 = x + N
                tiles = image[y:y+M,x:x+N]
                cv2.rectangle(image, (x, y), (x1, y1), (100, 100, 100))
        
        bboxes = np.reshape(bboxes,(10,bboxes.shape[1]*bboxes.shape[2]))
        # I concatenate the predictions from both bounding boxes in a (5,98) shaped array (p,cx_cell,cy_cell,w,h)
        
        concat_bboxes = np.concatenate((bboxes[:5,...],bboxes[5:10,...]),axis=1)
        nms_labels = np.reshape(labels, (20,labels.shape[1]*labels.shape[2]))
        
        filtered_conf_bboxes=[]
        filtered_labels = []
        initial_positions = []
        # I filter the boxes with low confidence and memorize the cell location of the boxes that pass the confidence threshold
        # this improves inference time by 3 times, because we don't go through all the boxes anymore
        # as the inference time scales by O(n^2) with n being number of boxes
        for i in range(concat_bboxes.shape[1]):
            if concat_bboxes[0,i] > conf_thresh:
                filtered_conf_bboxes.append(concat_bboxes[:5,i])
                if i < self.no_grids*self.no_grids:
                    filtered_labels.append(nms_labels[:20,i])
                else:
                    filtered_labels.append(nms_labels[:20,i-self.no_grids*self.no_grids])
                if i < self.no_grids*self.no_grids:
                    initial_positions.append(i)
                else:
                    initial_positions.append(i-self.no_grids*self.no_grids)
            else:
                continue
        filtered_conf_bboxes = np.asarray(filtered_conf_bboxes)
        filtered_labels = np.asarray(filtered_labels)
        
        # I apply Non-Maximum Suppression on all the bounding boxes predictions
        # it returns the boxes coordinates and the cell positions for the respective boxes
        nms_box_list, positions = self.nonmax_suppression(filtered_conf_bboxes,filtered_labels,initial_positions,nms_iou_cutoff)        
        
        for (i,box) in enumerate(nms_box_list):
            
            cx = box[1]
            cy = box[2]
            w_obj = box[3]
            h_obj = box[4]
            
            #get the 
            col = positions[i] % self.no_grids
            row = positions[i] // self.no_grids

            label_pos = np.argmax(labels[:20,row,col],axis=0)
            confidence_score = box[0]
            class_score = confidence_score* labels[label_pos,row,col] # I can either show the confidence score or the class score
            class_name = classes[label_pos]

            cX_imag = col/self.no_grids + cx/self.no_grids
            cY_imag = row/self.no_grids + cy/self.no_grids

            startX = int((cX_imag-w_obj/2)*w_img)
            startY = int((cY_imag-h_obj/2)*h_img)
            endX = int((cX_imag+w_obj/2)*w_img)
            endY = int((cY_imag+h_obj/2)*h_img)
            #print(startX,startY,endX,endY,cX_imag,cY_imag)
            
            # draw the predicted bounding
            # ng box and class label on the image
            y = startY - 10 if startY - 10 > 10 else startY + 10
            image = cv2.circle(image, (int(cX_imag*w_img),int(cY_imag*h_img)), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(image, "{}:{:.2f} ".format(class_name,class_score), (startX, y), cv2.FONT_HERSHEY_SIMPLEX,	0.65, (0, 255, 0), 2)
        return image

    
    def transform_from_grid(self,bboxes,labels,image):
        (h, w) = image.shape[:2]
        M = h//self.no_grids
        N = w//self.no_grids

        for y in range(0,h,M):
            for x in range(0, w, N):
                y1 = y + M
                x1 = x + N
                tiles = image[y:y+M,x:x+N]
                cv2.rectangle(image, (x, y), (x1, y1), (100, 100, 100))

        for B in range(2):
            cx = bboxes[B*5+1,...]
            cy = bboxes[B*5+2,...]
            w_obj = bboxes[B*5+3,...]
            h_obj = bboxes[B*5+4,...]
            for (row,i) in enumerate(bboxes[B*5,...]):
                for (col,j) in enumerate(i):
                    label_pos = np.argmax(labels[:20,row,col],axis=0)
                    class_score = j* labels[label_pos,row,col]
                    if class_score > 0.25:
                        class_name = classes[label_pos]
                        cX_imag = col/self.no_grids + cx[row,col]/self.no_grids
                        cY_imag = row/self.no_grids + cy[row,col]/self.no_grids

                        startX = int((cX_imag-w_obj[row,col]/2)*w)
                        startY = int((cY_imag-h_obj[row,col]/2)*h)
                        endX = int((cX_imag+w_obj[row,col]/2)*w)
                        endY = int((cY_imag+h_obj[row,col]/2)*h)
                        #print(startX,startY,endX,endY,cX_imag,cY_imag)
                        
                        # draw the predicted bounding box and class label on the image
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        image = cv2.circle(image, (int(cX_imag*w),int(cY_imag*h)), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                        cv2.putText(image, "{}: {}".format(class_name,class_score), (startX, y), cv2.FONT_HERSHEY_SIMPLEX,	0.65, (0, 255, 0), 2)
        return image

"""
imageDir = 'dataset/images'
annotDir = 'dataset/annotations'
data = []
bboxes = []
labels = []
no_objects = []
dataset = VOCdataset()
data,bboxes,labels,no_objects,imagePaths = dataset.load_dataset(imageDir,annotDir)
GT = GridTransform()
bboxes_t,labels_t = GT.transform(bboxes,labels,no_objects)
#Test transformare corecta
no_grids=9
for (step,imagePath) in enumerate(imagePaths):
    if step == 15:
        break
    else:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
        [h,w] = image.shape[:2]
        GT.transform_from_grid(bboxes_t[step,...],labels_t[step,...],h,w,image)
"""
