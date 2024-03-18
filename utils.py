from PIL import Image
from plot import *
import numpy as np
import matplotlib.pyplot as plt
import png
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def find_maxima(matrice, num_max, sigma):
    punti_massimi = []

    # Scansiona tutti i pixel della matrice
    for x in range(matrice.shape[0]):
        for y in range(matrice.shape[1]):
            valore = matrice[x, y]
            punti_massimi.append((x, y, valore, sigma))

    # Ordina i punti in base al valore in ordine decrescente
    punti_massimi = sorted(punti_massimi, key=lambda punto: abs(punto[2]), reverse=True)
    # Prendi solo i primi 100 punti di massimo ordinati
    punti_massimi = punti_massimi[:num_max]

    return  punti_massimi


def filter_max_points(punti_massimi):
    punti_filtrati = []

    for punto in punti_massimi:
        x1, y1, value1, sigma1 = punto
        # Controlla la points_distance con tutti i punti precedenti
        trovato = False

        for i, altro_punto in enumerate(punti_filtrati):
            x2, y2, value2, sigma2 = altro_punto
            points_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            min_dist = min(sigma1,sigma2)
                
            if points_distance < min_dist:
                trovato = True
                # Se il valore del punto corrente è maggiore, sostituisci il punto precedente
                if value1 > value2:
                    punti_filtrati[i] = punto
                
        # Se non è stato trovato un punto vicino con un valore minore, aggiungi il punto corrente
        if not trovato:
            punti_filtrati.append(punto)

    return punti_filtrati