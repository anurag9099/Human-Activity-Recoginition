#  ##################
#
# Load the required libraries
#
#  ##################

import os
import os.path
from keras.layers import LSTM,TimeDistributed
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
import cv2
import time
import pickle
from keras.callbacks import TensorBoard
from time import time
import pandas as pd
from matplotlib import pyplot as plt


#  ##################
#
#   get the names of the video files from the File directory
#   
#  ##################


names=os.listdir(r'C:\Users\Dell\Desktop\Projects\Humandetection\RGBdallas')


##############
#
# Isolate some of the videos for testing the performance of the model
#
############
test=[]
for i in range(0,len(names)):
    if 't4' in names[i]:
        test.append(names[i])
#############
#
#Get the videos apart from the test for model training
#
#################
train=[i for i in names if i  not in test]


#  ##################
#
#    Tensorflow API detection(which detects the human in the images)
#   
#
#  ##################

#Show the path of the model graph

model_path = r'C:\Users\Dell\Desktop\Projects\Humandetection\ssd_inception_v2_coco_11_06_2017\frozen_inference_graph.pb'
path_to_ckpt=model_path
#Set the threshold of detecting poerson to 0.6
threshold = 0.6

path_to_ckpt = r'C:\Users\Dell\Desktop\Projects\Humandetection\ssd_inception_v2_coco_11_06_2017\frozen_inference_graph.pb'
#Load the graph from the tensorflow file
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

default_graph = detection_graph.as_default()
sess = tf.Session(graph=detection_graph)

# Definite input and output Tensors for detection_graph
image_tensor =detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def processFrame(image):
    
    # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    # Actual detection.
    start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    end_time = time.time()


    im_height, im_width,_ = image.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[0,i,0] * im_height),
                    int(boxes[0,i,1]*im_width),
                    int(boxes[0,i,2] * im_height),
                    int(boxes[0,i,3]*im_width))

    return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

def close(self):
    self.sess.close()
    self.default_graph.close()


#  ##################
#
#     Load the images and their respective labels
#   
#  ##################


#Create two lists to store images and their respective labels
label_files=[]
images_files=[]
#Create a for loop to store the frames in the videos
for i in range(0,len(train)):
    #Get th label(action performed) from the video name
    x=train[i][:3]
    #Get the path of the video
    path=os.path.join(r'C:\Users\Dell\Desktop\Projects\Humandetection\RGBdallas' ,train[i])
    k=0
    #Open the video which is located in th path
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Load the frames from the  video till the video is open 
    while(cap.isOpened()):
        #get the grame from the video
        ret, frame = cap.read()
        if ret:
            k=k+1
            orig_image=frame
            #Get the boxes ,scores,classes from the Tensorflow API
            boxes, scores, classes, num = processFrame(frame)
            for l in range(len(boxes)):
                # Class 1 represents human and check if the score of the class is greater than threshold
                if classes[l] == 1 and scores[l] > threshold:
                    box = boxes[l]
                    #Crop the image with the boxes obtained from the Tensorflow API
                    image=orig_image[box[0]:box[2],box[1]:box[3]]
                    #Convert the image from colour to Gray
                    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #REsize the cropped image to 150 ,150
                    img_file = cv2.resize(gray_im, (150, 150))
                    #Store it in the form of array using Keras pre processing library
                    img_file = img_to_array(img_file)
                    #Convert each pixel value to float and normalize it to range of (0-1)
                    img_file = (img_file).astype("float") / 255.0
                    img_file = img_to_array(img_file)
                    #Append to the list we have initialised in the beginning
                    images_files.append(img_file)
                    #Append the respective label for the frame
                    label_files.append(x)

            
                
    print('Images loaded till now',len(images_files))     



# 
#  ##################
#
#     Dump the load images in the pickle file
#
#  ##################

#Create two files to store images and their labels in to pickle files


filename = "images_files_new1.dat"
with open(filename, 'wb') as f:
    pickle.dump(images_files,f)
print(1)
filename = "label_files_new1.dat"
with open(filename, 'wb') as f:
    pickle.dump(label_files,f)


# # ###############
#
# For loading pickle files
#
##############


filename = "images_files.dat"
with open(filename, 'rb') as f:
    images= pickle.load(f)
filename = "label_files.dat"
with open(filename, 'rb') as f:
    labels= pickle.load(f)






