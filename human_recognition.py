#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
from tkinter import filedialog
import dlib
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tkinter import ttk
from PIL import ImageTk, Image


# In[2]:


global index_list
global names
# Dictionary of Actions performed and their labels
cols={'a10':'right hand draw circle (counter clockwise)', 'a11':'draw triangle', 'a12':'bowling (right hand)', 'a13':'front boxing', 'a14':'baseball swing from right', 
      'a15':'tennis right hand forehand swing', 'a16':'arm curl (two arms)', 'a17':'tennis serve', 'a18':'two hand push', 'a19':'right hand knock on door',
       'a1_':'right arm swipe to the left', 'a20':'right hand catch an object', 'a21':'right hand pick up and throw', 'a22':'jogging in place', 'a23':'walking in place', 'a24':'sit to stand', 'a25':'stand to sit',
      'a26':'forward lunge (left foot forward)', 'a27':'squat (two arms stretch out)', 'a2_':'right arm swipe to the right',
       'a3_':'right hand wave', 'a4_':'two hand front clap', 'a5_':'right arm throw', 'a6_':'cross arms in the chest', 'a7_':'basketball shoot', 'a8_':'right hand draw x', 'a9_':' right hand draw circle (clockwise)'}
names=['a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
       'a1_', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a2_',
       'a3_', 'a4_', 'a5_', 'a6_', 'a7_', 'a8_', 'a9_']
# Load the model
global model
model = load_model('LSTM_new.h5')


# In[3]:


# Tesnorflow face detection
# Load the tensorflow file of the model
model_path = 'human-detection.pb'
path_to_ckpt=model_path
# Set the threshold of detection
threshold = 0.6

path_to_ckpt = 'human-detection.pb'
#Import the graph in to the directory
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

#Create a function to process the image
def processFrame(image):
    
    # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    # Actual detection.
    start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    end_time = time.time()

#         print("Elapsed Time:", end_time-start_time)

    im_height, im_width,_ = image.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[0,i,0] * im_height),
                    int(boxes[0,i,1]*im_width),
                    int(boxes[0,i,2] * im_height),
                    int(boxes[0,i,3]*im_width))

    return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])




def video_upload(root):
    filename = filedialog.askopenfilename(filetypes=[("Avi",".avi"),("Mp4",'.mp4')])
    global cam
    cam = cv2.VideoCapture(filename)
    global length
 
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label))
def quit_(root):
    cam.release()
    root.destroy()
    

def update_image(image_label, cam):
    #tree.delete(*tree.get_children())
    timedict={}
    start = time.time()
    num_frames = 0
    #model = load_model("model_nb.h5")
    while cam.isOpened():
        readsuccessful, f = cam.read()
        fps = cam.get(cv2.CAP_PROP_FPS)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        orig_image=f
        boxes, scores, classes, num = processFrame(f)
        for l in range(len(boxes)):
            # Class 1 represents human
            if classes[l] == 1 and scores[l] > threshold:
                box = boxes[l]
                img_file=orig_image[box[0]:box[2],box[1]:box[3]]
                #print(readsuccessful)
                t1_start = time.time()
                a1 = f

                gray_im = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
                img_file = cv2.resize(gray_im, (150, 150))
                img_file = img_to_array(img_file)
                img_file = (img_file).astype("float") / 255.0
                img_file = img_to_array(img_file)
                img_file = np.expand_dims(img_file, axis=0)
                img_file=np.reshape(img_file,(img_file.shape[0],150,150))
                global prob
                prob = model.predict(img_file)
                if prob[0][prob.argmax()]>0.65:
                    pred=cols[names[prob.argmax()]]
                    if pred in timedict:
                        timedict[pred]+=1
                    else:
                        timedict[pred]=1
                    my_label.config(text = pred)
        a = Image.fromarray(a1)
        b = ImageTk.PhotoImage(image=a)
        image_label.configure(image=b)
        image_label._image_cache = b  # avoid garbage collection
        root.update()
        num_frames+=1
        for i in tree.get_children():
            tree.delete(i)
        for k,v in timedict.items():
            tree.insert('', 'end', text=str(v),values=([k]))
        if num_frames==length:
            cam.release()
            break

    cam.release()    
            
def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    try:
        fps = int(float(count_of_deltas) / sum_of_deltas)
    except ZeroDivisionError:
        fps = 0
    fps_label.configure(text='FPS: {}'.format(fps))


def update_all(root, image_label, cam, fps_label):
    update_image(image_label, cam)
    update_fps(fps_label)
    root.after(20, func=lambda: update_all(root, image_label, cam, fps_label))


# In[5]:


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Facial action recognition")

    width, height = 800,600
    root.configure(background='#FFD933')
    root.geometry('%dx%d+0+0' % (width,height))
    image_label = tk.Label(master=root)# label for the video frame
    image_label.pack()
   
    
    fps_label = tk.Label(master=root)# label for fps
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    # Upload button
    upload_button = tk.Button(master=root, text='video upload',command=lambda: video_upload(root),fg="white",bg="blue",font=("Quicksand", 12))
    upload_button.pack(padx=5,pady=10,side='left')
    # quit button
    quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root),fg="white",bg="blue",font=("Quicksand", 12))
    quit_button.pack(padx=5,pady=10,side='right')
    
    my_label = tk.Label(root,text = "Output",font=("Quicksand", 16),bg='#FFD933',fg="blue")
    my_label.pack(padx=5,pady=10)

    tree = ttk.Treeview(root,columns=(['Action Performed']),selectmode="extended")
    tree.pack()
    tree.heading("#0",text='Count')
    tree.column("#0",width=100,anchor='center')
    tree.heading("#1",text='Action Performed')
    tree.column("#1",width=250,anchor='center')
    root.mainloop()


# In[ ]:





# In[ ]:




