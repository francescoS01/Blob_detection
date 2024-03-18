from PIL import Image
from plot import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PIL import Image


def gaussian_function(x,y,sigma):
    kernel_value = -1 / (np.pi * sigma ** 2) * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel_value


def filter_create(sigma):
    x = int(np.ceil(sigma * 6))
    # Creazione della griglia di coordinate centrata
    x_values = np.arange(-(x // 2), x // 2 +1)
    y_values = np.arange(-(x // 2), x // 2 +1)
    x, y = np.meshgrid(x_values, y_values)
    filter_size = len(x_values)
    
    # Calcolo dei valori dei pixel
    filter = gaussian_function(x, y, sigma)
    
    return filter, filter_size


# convolution of filter in the image 
def filter_convolution(image, filter, filter_size):
    
    # image dim
    image_row, image_col = image.shape[:2]
    
    # filter dim
    filter_col = filter_size
    filter_row = filter_size

    # filter_matrix (inizalize to 0)
    filter_matrix_row = (image_row - filter_row) +1
    filter_matrix_col = (image_col - filter_col) +1
    filter_matrix = np.zeros((filter_matrix_row, filter_matrix_col))

    for x in range(filter_matrix_row):
        for y in range(filter_matrix_col):
            filter_matrix[x, y] = np.sum(image[x:x + filter_size, y:y + filter_size] * filter)

    return filter_matrix