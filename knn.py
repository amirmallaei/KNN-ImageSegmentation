__author__ = "Amir Mallaei"
__email__ = "amirmallaei@gmail.com

import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy.matlib
import numpy as np

from Myfuncs import my_mean


# Read the Original Image
img = np.array(Image.open('1.jpg'))
size_img = img.shape

# Read the Masks
water = np.array(Image.open('water.jpg'))
green = np.array(Image.open('green.jpg'))
urban = np.array(Image.open('urban.jpg'))

# Define Classes
classes = ['water', 'green', 'urban']
nClasses = len(classes)
sample_regions = np.zeros((size_img[0], size_img[1], nClasses))

for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        if water[i, j] > 0:
            sample_regions[i, j, 0] = 1
        if green[i, j] > 0:
            sample_regions[i, j, 1] = 1
        if urban[i, j] > 0:
            sample_regions[i, j, 2] = 1

# Display sample regions
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.title.set_text('Original Image')
ax1.imshow(img)
ax2.title.set_text('Water Section')
ax2.imshow(sample_regions[:, :, 0], cmap='gray')
ax3.title.set_text('Greenary Section')
ax3.imshow(sample_regions[:, :, 1], cmap='gray')
ax4.title.set_text('Urban Section')
ax4.imshow(sample_regions[:, :, 2], cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
plt.show()

# Convert to L*a*b
lab_x = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

# Calculate the mean 'a*' and 'b*' value for each section
a = lab_x[:, :, 1]
b = lab_x[:, :, 2]
color_markers = np.zeros((nClasses, 2), dtype=float)

# Classify each pixel using the nearest neighbor
for i in range(0, nClasses):
    color_markers[i, 0] = my_mean(sample_regions[:, :, i], a)
    color_markers[i, 1] = my_mean(sample_regions[:, :, i], b)

# Perform classification
distance = np.zeros(size_img)
for i in range(0, nClasses):
    distance[:, :, i] = (np.power((a - color_markers[i, 0]), 2) +
                         np.power((b - color_markers[i, 1]), 2))

output = np.zeros(size_img)
value = np.zeros((size_img[0], size_img[1]))
label = value
for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        if distance[i, j, 0] < distance[i, j, 1] and distance[i, j, 0] < distance[i, j, 2]:
            output[i, j, 2] = 255
        elif distance[i, j, 1] < distance[i, j, 0] and distance[i, j, 1] < distance[i, j, 2]:
            output[i, j, 1] = 255
        elif distance[i, j, 2] < distance[i, j, 1] and distance[i, j, 2] < distance[i, j, 0]:
            output[i, j, 0] = 255

# Display Results
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
ax1.title.set_text('Water Section')
ax1.imshow(output[:, :, 2], cmap='gray')
ax2.title.set_text('Greenary Section')
ax2.imshow(output[:, :, 1], cmap='gray')
ax3.title.set_text('Urban Section')
ax3.imshow(output[:, :, 0], cmap='gray')
ax4.imshow(output)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
plt.show()
