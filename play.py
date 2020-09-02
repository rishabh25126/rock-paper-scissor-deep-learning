import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import cv2
import os


MAP_CLASS = {
    0 : "rock",
    1: "paper",
    2: "scissor",
    3:"none"
}

# Load Model
model = tf.keras.models.load_model('rock-paper-scissors-model.h5')

# Capture Video
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
     # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    image = cv2.rectangle(gray, (5, 5) , (220, 220) , (255, 0, 0) , 2)  

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        start = True
        image_model = image[5:220,5:220]
        image_model = cv2.resize(image_model, (50, 50))
        image_model = image_model.reshape(1,50,50,1)
        pred = model.predict(image_model)
        findd = np.array(pred[0])
        result = np.where(findd == max(findd))
        print("Result: ", MAP_CLASS[result[0][0]])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()