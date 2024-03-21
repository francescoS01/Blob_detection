from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
import png
from matplotlib.patches import Circle
from plot import*
from utils import*
from convolution import*


def blob_detection(sigmas, image_path, num_max, image_name): 
    """
    executes blob detection on an image using the Laplacian of Gaussian (LoG) method.

    INPUT:
    sigmas (list): List of sigma values to use for creating LoG filters.
    image_path (str): Path of the image on which to perform blob detection.
    num_max (int): Maximum number of points of interest to be detected for each sigma scale.

    OUTPUT:
    No directly returned value. The function creates a plot of the input image with detected blobs surrounded by circles.
    """

    #image
    im = Image.open(image_path) # open the image
    image = np.array(im)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # array taht contain the points of interest
    max_points = []

    for sigma in sigmas:

        # create the filter and take its dimention 
        filter, filter_size = filter_create(sigma)
        # plot(filter_size, filter, "my_filter")

        # do the padding to maintain the coordinates
        image_padding = np.pad(gray_image, int((filter_size-1)/2), mode='edge')

        # convolution of filter on the image
        filter_image = filter_convolution(image_padding, filter, filter_size)

        # take num_max max of the filter image with fixed sigma 
        max_points += find_maxima(filter_image, num_max, sigma)        
    

    # points that are very close to each other with respect to sigma are eliminated
    max_points = filter_max_points(max_points.copy())
    #max_points = filter_max_points(max_points)

    #print(filter_max_points)

    # the points of interest are circled and the plot is made
    blob_detection_plot(gray_image, max_points, image_name)



"""
sigmas = [1, 2, 4, 8, 16, 32]
image_path = "horse160.png"
num_max = 100 # number of maximum points that will be taken for every sigma

blob_detection(sigmas, image_path, num_max)

"""

sigmas = [2, 4, 8, 16, 32]
image_path = "image/horse024.png"
image_name = "horse024.png"
num_max = 400

blob_detection(sigmas, image_path, num_max, image_name)


# uno buono = 400
# da tenere abbiamo 17 - 21 - 24 - 43 
#prova