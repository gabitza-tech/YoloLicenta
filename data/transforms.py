from data.data import VOCdataset
import argparse
import numpy as np

classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

class GridTransform:
    def __init__(self,B=2,no_grids=11):
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
                bboxes_grid[1][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][0]
                bboxes_grid[2][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][1]
                bboxes_grid[3][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][2]
                bboxes_grid[4][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = cell_centers_image[i][j][3]

                label_grid[labels[i][j]][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1
                bboxes_grid[0][cell_pos_image[i][j][1]][cell_pos_image[i][j][0]] = 1 ## has_obj matrix

            self.bboxes_tr.append(bboxes_grid)
            self.labels_tr.append(label_grid)

        #print(cell_pos_image[1])
        #print(self.bboxes_tr[1])
        #print(self.labels_tr[1])
        self.bboxes_tr = np.array(self.bboxes_tr,dtype='float32')
        self.labels_tr = np.array(self.labels_tr,dtype='float32')
        return self.bboxes_tr,self.labels_tr

""""
imageDir = 'dataset/images'
annotDir = 'dataset/annotations'

data = []
bboxes = []
labels = []
no_objects = []

dataset = VOCdataset()
data,bboxes,labels,no_objects,imagePaths = dataset.load_dataset(imageDir,annotDir)

print(data.shape)
print(bboxes[1])
print(labels[1])
print(no_objects[1])

GT = GridTransform()

bboxes_t,labels_t = GT.transform(data,bboxes,labels,no_objects)

#print(bboxes_t.shape)
#print(labels_t.shape)

print(bboxes_t[1,0:4,...])

"""