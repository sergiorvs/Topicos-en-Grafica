import numpy as np
import cv2
from PIL import Image
import math 
from matplotlib import pyplot as plt
import glob

# list_images = glob.iglob("letters/*")

# for image_title in list_images:



img = cv2.imread("/home/sergio/Descargas/mr.jpg", cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)

cv2.imshow("/home/sergio/Descargas/mr.jpg", img_and_magnitude)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

# import math
# import cmath

# #escala de grises
# import numpy as np
# import time
# from PIL import Image
# import copy



# T = Image.open("/home/sergio/Documentos/fourier7.jpg").convert('L')
# T.show()
# n = np.size(T, 0)
# m = np.size(T, 1)
# yNT = [[complex(0) for x in range(n)] for y in range(m)]
# mNT = [[complex(0) for x in range(n)] for y in range(m)]

# G = [  1,4,7,4,1,
#         4,16,26,16,4,
#         7,26,41,26,7,
#         4,16,26,16,4,
#         1,4,7,4,1]


# def Fourier(T):
#     T=np.array(T)
#     mN= np.zeros_like(T, np.uint32)
#     #agregar mascara a la imagen
#     for z in range(0,5):
#         for y in range(0,5):
#             mN[z][y] = G[z*5+y]
#     print mN
#     yN= np.zeros_like(T, np.uint32)

#     for u in range(0, n-1): #N
#         for v in range(0, m-1): #M
#             sumT = complex(0)
#             sumM = complex(0)
#             for x in range(0, n): #N
#                 for y in range(0, m): #M
#                     e = cmath.exp(-1j * math.pi * 2.0 * ((float(u*x) / n) + (float(v*y) / m)))
#                     sumT += e * complex(T[x][y])
#                     sumM += e * complex(mN[x][y])
#             yNT[u][v] = (sumT/(n*m))
#             yN[u][v] = (sumT.real/(n*m))
#             mNT[u][v] = (sumM/(n*m))
#             mN[u][v] = (sumM.real/(n*m))

#     img = Image.fromarray(yN)
#     img.show()

# Fourier(T)

