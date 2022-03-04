from data.data import VOCdataset
import argparse
import numpy as np
import cv2

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
                pos_cell_x = int(bbox[0]/self.w_cell)
                pos_cell_y = int(bbox[1]/self.h_cell)
                relative_cx = (bbox[0]-pos_cell_x*self.w_cell)/self.w_cell
                relative_cy = (bbox[1]-pos_cell_y*self.h_cell)/self.h_cell
                cell_pos.append([pos_cell_x,pos_cell_y])
                cell_centers.append([relative_cx, relative_cy, bbox[2], bbox[3]])
            cell_pos_per_image.append(cell_pos)
            cell_centers_per_image.append(cell_centers)
        return cell_pos_per_image, cell_centers_per_image
    
    def transform(self,data,bboxes,labels,no_objects):
        cell_pos_image, cell_centers_image = self.cell_pos_relative(bboxes)
        
        for i in range(len(no_objects)):
            bboxes_grid = np.zeros((self.B*5,self.no_grids,self.no_grids))
            label_grid = np.zeros((len(classes),self.no_grids,self.no_grids))
            for j in range(no_objects[i]):
                
                bboxes_grid[1][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][0]#cx
                bboxes_grid[2][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][1]#cy
                bboxes_grid[3][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][2]#w
                bboxes_grid[4][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][3]#h

                label_grid[labels[i][j]][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1 # label
                bboxes_grid[0][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1 ## has_obj matrix


            self.bboxes_tr.append(bboxes_grid)
            self.labels_tr.append(label_grid)

        #print(cell_pos_image[1])
        #print(self.bboxes_tr[1])
        #print(self.labels_tr[1])
        self.bboxes_tr = np.array(self.bboxes_tr,dtype='float32')
        self.labels_tr = np.array(self.labels_tr,dtype='float32')
        return self.bboxes_tr,self.labels_tr
    
    def transform_from_grid(self,bboxes,labels,h,w,image):
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
                    if j > 0.1:

                        label_pos = np.argmax(labels[:20,row,col],axis=0)
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
                        cv2.putText(image, class_name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,	0.65, (0, 255, 0), 2)
        cv2.imshow('da',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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

bboxes_t,labels_t = GT.transform(data,bboxes,labels,no_objects)


#Test transformare corecta

no_grids=9
for (step,imagePath) in enumerate(imagePaths):
    if step == 15:
        break
    else:
        image = cv2.imread(imagePath)
        [h,w] = image.shape[:2]
        GT.transform_from_grid(bboxes_t[step,...],labels_t[step,...],h,w,image)

"""
