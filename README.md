# YOLOv1 in Tensorflow 2.4 and Keras 2.6
The YOLOv1 algorithm implemented in Tensorflow 2.4 and Keras 2.6, with a network architecture consisting of a pretrained network and a fully connected head from Keras applications architectures. Image augmentation was not used in this project and for good validation/test results it is needed to use image augmentation. In the future, image augmentation will be implemented.
# YoloLicenta
Weights trained on ResNet50 with the YoloLoss loss from losses.py: https://drive.google.com/file/d/1NwRcqHB_dGR5gqqtt_YHrHgSwbqGOObj/view?usp=sharing
The hdf5 file should be placed in output folder.
# data/data.py
This file contains the VOCdataset class with the load_dataset() function. This function returns the boxes, labels, number of objects, image_path for each image. The dataset directory should contain a directory with images and a directory only with xml annotations. Each annotation is read independently and there are no csv files used.
# data/transforms.py
Contains multiple functions. First the transform() function that transforms the information for each image into a yolo format array of size 30,7,7. If you want to train the algorithm on a different dataset with different classes, the values for the variables must be updated. (Mostly where there is a number with 20, should be replaced with the new number of classes)
# predict_pascal.py
Choose to visualize prediction with transform_from_grid.py without nms, or with trasnform_with_nms. Model can be loaded through load_model function, preprocessing for predictions should be adapted to the trained model preprocessing.
# train.py
If input size is changed from (224,224), the W and H in yolo_loss should also be changed. When changing network, also change preprocessing. When changing the number of grids, also change the number of grids for yolo_loss, respectively S. 
# yolo_loss.py
Update W and H when using different input size than (224,224), also update S when using other number of grids than 7.

This application was created as a bachelor thesis project and as an education tool. Even though the code is thought for 20 classes, with minor tweekings it can be adapted for new datasets. There are only 2 bounding box predictions per cell, but this can also be change by adapting the B variable and also changing some parts in the transforms file and the loss files.
