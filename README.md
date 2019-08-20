# voice_activated_filters
***Note***: This is a modification of the code written by Akshay Chandra Lagandula: https://github.com/akshaychandra21/Selfie_Filters_OpenCV. His repository contains additional files which create a neural network for facial keypoint detection and train it. This repository simply loads the resulting model and uses it for keypoint detection.

Chad Bloxham. Done as the final project for a graduate course in neural networks. 
Group partners: Mehul Kothari, Srishti Tomar, Akshit Goyal.

## Improvements
Uses voice activation to change between facial filters. Includes several hat and mustache filters in addition to sunglasses.

## Run Instructions
***Requires***: an internet connection, Python 3.6.x (won't work for 3.7.x), Keras, Tensorflow, opencv-python, numpy, speech_recognition, threading.

Run filters.py. Filter 1 will be displayed. To change to another filter, say into microphone: "1", "2", "3", "4", or "5". There is a short lag between saying the filter and it being displayed. Any other audio input will cause the program to terminate, as will being too close to the webcam.

## filters.py Output:

<a href="https://imgflip.com/gif/38c6ly"><img src="https://i.imgflip.com/38c6ly.gif" title=""/></a>
