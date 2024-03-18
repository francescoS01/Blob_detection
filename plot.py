from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def filter_plot(size, imm, s): # Determina i limiti degli assi x e y in modo che il punto (0, 0) sia al centro del plot
    x_limit = size // 2
    y_limit = size // 2

    # Creazione del plot del filtro colorato in scala di grigi
    plt.imshow(imm, cmap='gray', extent=(-x_limit, x_limit, -y_limit, y_limit))  
    plt.title('Filtro con centro in (0, 0)')
    plt.colorbar(label='Valore del filtro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)  # Rimuove la griglia
    plt.savefig(s)  
    plt.show()


def blob_detection_plot(image, points, image_path):
    """
    Creates a plot of the original image with detected blobs surrounded by circles.

    INPUT:
    image: the image on which blobs were detected.
    points (list): List of points detected during blob detection. 
                   Each point is represented as a tuple (x, y, value, sigma)
                   x and y: are the coordinates of the point 
                   value: is the value associated with the point
                   sigma: is the sigma value used for detection.
    image_path (str): Path of the image used for blob detection.

    OUTPUT:
    No directly returned value. The image with detected blobs surrounded by circles is saved as a .png file.
    """

    # Create a copy of image
    image_copiata = np.copy(image)
    
    fig, ax = plt.subplots()
    ax.imshow(image_copiata, cmap='gray')
    ax.axis('off')
    
    for punto in points:
        x, y, _, sigma = punto
        # draw a circle with radius sigma on the original image for each point
        circle = Circle((y, x), radius=sigma, color='red', fill=False)
        ax.add_patch(circle)

    # save the image with circle in a file .png
    #save_path = 'result/BD_' + os.path.basename(image_path)
    s = "result/BD_" + image_path
    plt.savefig(s, bbox_inches='tight', pad_inches=0)
    plt.close()