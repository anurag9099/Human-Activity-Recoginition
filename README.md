# Human-Activity-Recoginition

Data Source: UTD-MHAD
Dataset was collected as part of research on human action recognition using fusion of depth and inertial sensor data. The objective of this research has been to develop algorithms for more robust human action recognition using fusion of data from differing modality sensors.

The goal of this project is to recognize and predict human actions from videos. A regular deep neural network would not suit this task for two main reasons: First, deep neural networks do not support handling translation or other distortions of the input, which happen frequently in action videos. Second, deep feed-forward neural networks do not have a state, therefore making processing of videos difficult as they require handling of states in order to recognize or predict actions.

### Prosposed Model

![Model flowchart](extras/flow.PNG)

For feature extraction, a CNN is trained on multiple frames of the entire video and on each frame of an area which suitable for human action detection, such as the swipe left, swipe right. This CNN is followed by one or multiple layers of a regular (dense) neural network for discrimination of the features.
    The output of this network is fed into the second part, which handles temporal relationships. Without temporal part, classifications are not up to the mark by only CNN. There are different possibilities how to model the temporal part. On the one hand, state-of-the-art methods, Recurrent Neural Network, could be used. On the other hand, Multilayer Perceptron is of particular interests for this project. LSTMs are reported to perform well on temporal data and MLP performed way better than LSTMs. Therefore, MLP are chosen for this part, followed by one or multiple layers of a regular neural network for discrimination of the features.


