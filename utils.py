import heapq
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
    heap = []

    # Scansiona tutti i pixel della matrice
    for x in range(matrice.shape[0]):
        for y in range(matrice.shape[1]):
            valore = matrice[x, y]
            
            # Mantieni solo i primi num_max valori massimi
            if len(heap) < num_max:
                heapq.heappush(heap, (abs(valore), x, y, valore, sigma))
            else:
                # Se il valore corrente è maggiore del minimo nel heap, sostituiscilo
                heapq.heappushpop(heap, (abs(valore), x, y, valore, sigma))

    # Converte gli elementi del heap in una lista ordinata
    punti_massimi = [(x, y, valore, sigma) for (_, x, y, valore, sigma) in heap]

    return punti_massimi


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

            if points_distance < 2*min_dist:
                trovato = True
                # Se il valore del punto corrente è maggiore, sostituisci il punto precedente
                if value1 > value2:
                    punti_filtrati[i] = punto
                break

        # Se non è stato trovato un punto vicino con un valore minore, aggiungi il punto corrente
        if not trovato:
            punti_filtrati.append(punto)
            

    return punti_filtrati