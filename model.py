#!/usr/bin/env python
# coding: utf-8

from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,Reshape,Input
from keras import applications
import os
import os.path
#from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.layers.core import Activation
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


# # Get the file names of videos from the directory

# In[2]:


names=os.listdir(r'RGB')
test=[]
for i in range(0,len(names)):
    if 't4' in names[i]:
        test.append(names[i])
train=[i for i in names if i  not in test]


# # Tesnorflow API for Human  detection

# In[3]:


model_path = 'human-detection.pb'
path_to_ckpt=model_path
threshold = 0.6

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
    #start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    #end_time = time.time()

#         print("Elapsed Time:", end_time-start_time)

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


# # Extracting features from  VGG

# In[4]:


model = VGG16()
# re-structure the model
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

images_files=[]
label_files=[]

for i in range(0,len(train)):
    x=train[i][:3]
    path=os.path.join(r'RGB' ,train[i])
    k=0
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            k=k+1
            orig_image=frame
            if k in range(round(length/2)-5,round(length/2)+5):
                boxes, scores, classes, num = processFrame(frame)
                for l in range(len(boxes)):
                    # Class 1 represents human
                    if classes[l] == 1 and scores[l] > threshold:
                        box = boxes[l]
                        image=orig_image[box[0]:box[2],box[1]:box[3]]
                        img_file = cv2.resize(image, (224, 224))
                        # convert the image pixels to a numpy array
                        image = img_to_array(img_file)
                        # reshape data for the model
                        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                        # prepare the image for the VGG model
                        image = preprocess_input(image)
                        
                        # get features
                        feature = model.predict(image, verbose=0)
                        #gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        images_files.append(feature)
                        label_files.append(x)
                        
        else:
            break     
                
    print('Images loaded :',len(images_files))     

# save to file
dump(images_files, open('VGG_net_features.pkl', 'wb'))
dump(label_files, open('VGG_net_Labels.pkl', 'wb'))


# # Loading the pickle file

all_features=[]
labels=[]
from pickle import load
for i in range(1,5):
    all_features.append(load(open(r'VGG_net_features.pkl', 'rb')))
    labels.append(load(open(r'VGG_net_Labels.pkl', 'rb')))


# # Converting the extracted features to CNN format

all_features1 = [item for sublist in all_features for item in sublist]
labels1 = [item for sublist in labels for item in sublist]

all_features=[item[0] for item in all_features1]
labels=np.array(pd.get_dummies(labels1))


# # Splitting in to train and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(all_features), labels, test_size=0.20, random_state=42)


# # Multi layer perceptron

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(27, init='uniform'))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.7, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=128,validation_split=0.2,verbose=1)


############### Save the model #######################
model.save('cnn+mlp_model.h5')

######### Model Evaluation on testing data ###########
model.evaluate(X_test,y_test)


######### storing the prediction of test data for AUC or ROC curve #####
y_score = model.predict_proba(testX)


######### building the AUC and ROC curve ########33
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc


n_classes=27

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(0)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','black','gold','green','red','darkred','navy','orangered'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend(loc="lower right")
plt.savefig('ModelGraph.png', dpi=360)
plt.show()