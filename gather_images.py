# Import libraries 
import numpy as np
import cv2
from time import sleep



# Capture Video
cap = cv2.VideoCapture(0)


start = False
count = 1


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
    if start:        
            # Display the resulting frame
            cv2.imshow('frame',image)

            #Saving the File
            filename = "images/" + str(count) + ".jpg"
            cv2.imwrite(filename, image[5:220,5:220])
            print(count)
            count+=1
    if count==501:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()