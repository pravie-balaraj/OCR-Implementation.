import cv2
import matplotlib.pyplot as plt
import numpy as np

# read the image and define the stepSize and window size 
# (width,height)
image = cv2.imread(r'\Users\pravi\OneDrive\Desktop\ML4\detection-images\detection-images\detection-1.jpg') # your image path
tmp = image # for drawing a rectangle
stepSize = 50
(w_width, w_height) = (20, 20) # window size
for x in range(0, image.shape[1] - w_width , stepSize):
   for y in range(0, image.shape[0] - w_height, stepSize):
      window = image[x:x + w_width, y:y + w_height, :]


      # draw window on image
cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 1) # draw rectangle on image
plt.imshow(np.array(tmp).astype('uint8'))
# show all windows
plt.show()