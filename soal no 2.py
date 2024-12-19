import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
original_image = iio.imread(file_name)

def rgb2gray(image):
    if len(image.shape) == 3:  
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) 
    return image

gray_image = rgb2gray(original_image)

def histogram_equalization(image):
    image_flattened = image.flatten()

    hist, bins = np.histogram(image_flattened, bins=256, range=[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  

    equalized = np.interp(image_flattened, bins[:-1], cdf_normalized * 255)
    return equalized.reshape(image.shape)

enhanced_image = histogram_equalization(gray_image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Enhanced Image")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.show()
