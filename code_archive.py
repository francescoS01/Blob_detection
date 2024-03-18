from PIL import Image
from plot import *
import numpy as np
import matplotlib.pyplot as plt
import png
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def write_png(filename, pixels, width, height, metadata):
    # Scrive l'immagine PNG
    with open(filename, 'wb') as file:
        writer = png.Writer(width=width, height=height, **metadata)
        writer.write(file, pixels)


def read_png(filename):
    # Open and read the PNG file PNG
    with open(filename, 'rb') as file:
        reader = png.Reader(file)
        width, height, pixels, metadata = reader.read()
        pixel_data = list(pixels)
        image = Image.open(filename)
        # obtain the dimention of immage
        width, height = image.size
        return [list(row) for row in pixel_data], width, height, metadata
    
def disegna_punti_massimo(image, punti_massimo):
    # Crea una copia dell'immagine originale in scala di grigi
    image_copiata = cv2.cvtColor(cv2.convertScaleAbs(image), cv2.COLOR_GRAY2BGR)

    # Disegna un cerchio rosso intorno a ciascun punto di massimo
    for punto in punti_massimo:
        x, y, value, sigma = punto
        # Converti sigma in un intero
        sigma_int = int(sigma)
        # Inverti le coordinate x e y e passale come una tupla
        cv2.circle(image_copiata, (y, x), radius=sigma_int, color=(0, 0, 255), thickness=2)

    return image_copiata

def disegna_punti_massimo1(image, punti_massimo):
    # Disegna l'immagine
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Disabilita gli assi
    
    # Disegna un cerchio rosso intorno a ciascun punto di massimo
    for punto in punti_massimo:
        x, y, value, sigma = punto
        # Converti sigma in un intero
        sigma_int = int(sigma)
        # Disegna il cerchio intorno al punto
        circle = plt.Circle((y, x), radius=sigma_int, color='red', fill=False)
        plt.gca().add_patch(circle)
    
    # Salva l'immagine in un array numpy
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Rimuove spazi bianchi intorno all'immagine
    plt.savefig("image_with_points.png", bbox_inches='tight', pad_inches=0)  # Salva l'immagine come PNG
    plt.close()  # Chiudi la figura per liberare memoria
    
    # Carica l'immagine salvata e restituiscila come array numpy
    image_with_points = plt.imread("image_with_points.png")
    
    return image_with_points


# -------------- PLOT -----------------

def plot_image(image, s):
    """
    function that plots the image and saves it as PNG file
    
    :param image: image to plot
    :param sigma: standard deviation of the filter
    :param s: filename to save the image
    """
    plt.imshow(image, cmap='gray')
    plt.title(s)
    plt.axis('off')  # Disabilita gli assi
    plt.savefig(s, bbox_inches='tight', pad_inches=0)  # Salva l'immagine come PNG
    plt.show()


def plot2(imm, s):
    # Ottieni le dimensioni dell'immagine
    height, width = imm.shape

    # Determina i limiti degli assi x e y in modo che il punto (0, 0) sia al centro del plot
    x_limit = width // 2
    y_limit = height // 2

    # Creazione del plot del filtro colorato in scala di grigi
    plt.imshow(imm, cmap='gray', extent=(-x_limit, x_limit, -y_limit, y_limit))  
    plt.title('Filtro con centro in (0, 0)')
    plt.colorbar(label='Valore del filtro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)  # Rimuove la griglia
    plt.savefig(s)  
    plt.show() 
    

def create_image_from_matrix(matrix, output_path):
    # Normalizza i valori della matrice tra 0 e 255
    matrix = np.clip(matrix, 0, 255).astype(np.uint8)
    # Crea un'immagine PIL utilizzando la matrice
    image = Image.fromarray(matrix)
    # Salva l'immagine su disco
    image.save(output_path)



"""

# ---------TEST exact image filter ---------

img = Image.open(image_path)

# Apply Gaussian blur with sigma 
blurred = cv2.GaussianBlur(np.array(img), (0, 0), sigma)

# Apply Laplacian of Gaussian (LoG) filter
result1 = cv2.Laplacian(blurred, cv2.CV_64F)
result1 = np.uint8(np.absolute(result1))

plot_image(result1, "result1.png")

# Salva l'immagine risultante come file PNG nella directory corrente
#output_image_path = "exact_out_im.png"
#cv2.imwrite(output_image_path, result1)


print("Immagine risultante salvata correttamente come")

"""



"""
def gaussian_function2(x,y,sigma): 
    z = -(x**2 + y**2) / (2*(sigma**2))
    return -(1/((sigma**2)*(sigma**2)))*(1+z)*np.exp(z)*(sigma**2)

def gaussian_function1(x, y, sigma):
    coefficient = -1 / (np.pi * sigma**4)
    exponent = (x**2 + y**2) / (2 * sigma**2)
    return (coefficient * (1 - exponent) * np.exp(-exponent))

    
    
    
# ------------- for TEST exact filter -----------------
def LoG(sigma):
    #window size 
    n = np.ceil(sigma*6)
    print(n)
    y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = sigma**2*(-(2*sigma*2) + (x*x + y*y) ) *  (x_filter*y_filter) * (1/(2*np.pi*sigma*4))
    return final_filter, n

"""