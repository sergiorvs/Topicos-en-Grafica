import numpy as np
import cv2
from PIL import Image
import math 
from matplotlib import pyplot as plt


# HISTOGRAMA

def drawHistogram(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    size = rows*cols
    cv2.imshow('Original', img)
    cv2.waitKey()
 



    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # print("histograma", hist)
    plt.plot(hist, color = 'gray')
    plt.show()

    # print(hist)

    sks = []
    sk = 0
    for i in range(0, len(hist)):
        sk += (hist[i]/size)
        to_add = sk*255
        sks.append(int(round(to_add, 0)))
    
    # print(sks)
    plt.plot(sks, color = 'red')
    plt.show()

    ecualized_image = img.copy()
    for i in range(0, rows):
        for j in range(0, cols):
            ecualized_image[i][j] = sks[ ecualized_image[i][j] ]

    # print(ecualized_image)
    cv2.imshow('Ecualized Image', ecualized_image)
    cv2.waitKey()



# FILTROS

def media(image_path, tam):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)
    aux_x = (tam*tam)/2
    aux_x = aux_x/tam
    aux_y = aux_x%tam

    for i in range(0, rows-tam):
        for j in range(0, cols-tam):
            val = 0
            for x in range(0, tam):
                for y in range(0, tam):
                    if(i+x<rows and j+x<cols):
                        val += img[i+x][j+y]
            val = val/(tam*tam)
            if(i+tam < rows and j+tam<cols ):
                new_image[i+aux_x][j+aux_y] = val
            # new_image[i][j] = (int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + 
            #                    int(img[i-1][j]) + int(img[i][j]) + int(img[i+1][j]) + 
            #                    int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/9
    
    cv2.imshow('Media', new_image)
    cv2.waitKey()

def media_ponderada(image_path, peso):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = (int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + 
                               int(img[i-1][j]) + peso*int(img[i][j]) + int(img[i+1][j]) + 
                               int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/(8+peso)
    
    cv2.imshow('MediaPonderada', new_image)
    cv2.waitKey()

def mediana(image_path, tam):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            ds = []
            for x in range(0, tam):
                for y in range(0, tam):
                    ds.append(img[i+x][j+y])
            # ds.append(img[i-1][j-1])
            # ds.append(img[i][j-1])
            # ds.append(img[i+1][j-1])
            # ds.append(img[i-1][j])
            # ds.append(img[i][j])
            # ds.append(img[i+1][j])
            # ds.append(img[i-1][j+1])
            # ds.append(img[i][j+1])
            # ds.append(img[i+1][j+1])

            ds = sorted(ds)
            # print(ds)
            new_image[i][j] = ds[5]
    
    cv2.imshow('Mediana', new_image)
    cv2.waitKey()

def maxmin_max(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            ds = []
            ds.append(img[i-1][j-1])
            ds.append(img[i][j-1])
            ds.append(img[i+1][j-1])
            ds.append(img[i-1][j])
            ds.append(img[i][j])
            ds.append(img[i+1][j])
            ds.append(img[i-1][j+1])
            ds.append(img[i][j+1])
            ds.append(img[i+1][j+1])

            ds = sorted(ds)
            new_image[i][j] = ds[8]
    
    cv2.imshow('MaxMin_MAX', new_image)
    cv2.waitKey()

def maxmin_min(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            ds = []
            ds.append(img[i-1][j-1])
            ds.append(img[i][j-1])
            ds.append(img[i+1][j-1])
            ds.append(img[i-1][j])
            ds.append(img[i][j])
            ds.append(img[i+1][j])
            ds.append(img[i-1][j+1])
            ds.append(img[i][j+1])
            ds.append(img[i+1][j+1])

            ds = sorted(ds)
            new_image[i][j] = ds[0]
    
    cv2.imshow('MaxMin_MIN', new_image)
    cv2.waitKey()

def gauss(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = (int(img[i-1][j-1]) + 2*int(img[i][j-1]) + int(img[i+1][j-1]) + 
                               2*int(img[i-1][j]) + 4*int(img[i][j]) + 2*int(img[i+1][j]) + 
                               int(img[i-1][j+1]) + 2*int(img[i][j+1]) + int(img[i+1][j+1]))/16
    
    cv2.imshow('Gauss', new_image)
    cv2.waitKey()


#BORDES

def derivada1(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = int(img[i][j+1]) - int(img[i][j])
    
    cv2.imshow('derivada_x', new_image)
    cv2.waitKey()  

def derivada2(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = int(img[i][j+1]) + int(img[i][j-1]) - 2*int(img[i][j])
    
    cv2.imshow('derivada2_x', new_image)
    cv2.waitKey()  

def laPlace(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i][j+1]) + int(img[i][j-1]) + int(img[i+1][j]) + int(img[i-1][j]) - 4*int(img[i][j])
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    
    cv2.imshow('laPlace', new_image)
    cv2.waitKey()     

def roberts(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sumx = -1*int(img[i-1][j-1]) + int(img[i][j])
            if(sumx > 255):
                Gx[i][j] = 255
            elif(sumx < 0):
                Gx[i][j] = 0
            else:
                Gx[i][j] = sumx

            sumy = -1*int(img[i-1][j]) + int(img[i][j-1])
            if(sumy > 255):
                Gy[i][j] = 255
            elif(sumy < 0):
                Gy[i][j] = 0
            else:
                Gy[i][j] = sumy
    
    new_image = Gx + Gy 
    cv2.imshow('Gx', Gx)
    cv2.imshow('Gy', Gy)
    cv2.imshow('Roberts', new_image)
    cv2.waitKey()     
        
def sobel(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sumx = int(img[i-1][j-1]) - int(img[i-1][j+1]) + 2*int(img[i][j-1]) - 2*int(img[i][j+1]) + int(img[i+1][j-1]) - int(img[i+1][j+1])
            if(sumx > 255):
                Gx[i][j] = 255
            elif(sumx < 0):
                Gx[i][j] = 0
            else:
                Gx[i][j] = sumx

            sumy = int(img[i-1][j-1]) - int(img[i+1][j-1]) + 2*int(img[i-1][j]) - 2*int(img[i+1][j]) + int(img[i-1][j+1]) - int(img[i+1][j+1])
            if(sumy > 255):
                Gy[i][j] = 255
            elif(sumy < 0):
                Gy[i][j] = 0
            else:
                Gy[i][j] = sumy

    new_image = Gx + Gy 
    cv2.imshow('Gx', Gx)
    cv2.imshow('Gy', Gy)
    cv2.imshow('Sobel', new_image)
    cv2.waitKey()  

def prewitt(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sumx = -1*int(img[i-1][j-1]) + int(img[i-1][j+1]) - int(img[i][j-1]) + int(img[i][j+1]) - int(img[i+1][j-1]) + int(img[i+1][j+1])
            if(sumx > 255):
                Gx[i][j] = 255
            elif(sumx < 0):
                Gx[i][j] = 0
            else:
                Gx[i][j] = sumx
            
            sumy = -1*int(img[i-1][j-1]) + int(img[i+1][j-1]) - int(img[i-1][j]) + int(img[i+1][j]) - int(img[i-1][j+1]) + int(img[i+1][j+1])
            if(sumy > 255):
                Gy[i][j] = 255
            elif(sumy < 0):
                Gy[i][j] = 0
            else:
                Gy[i][j] = sumy
    
    new_image = Gx + Gy 
    cv2.imshow('Gx', Gx)
    cv2.imshow('Gy', Gy)
    cv2.imshow('PreWitt', new_image)
    cv2.waitKey()  

if __name__ == "__main__":
    # path = "/home/sergio/TCG/PruebasImagenes/lenaS.png"

    # path2 = "/home/sergio/TCG/PruebasImagenes/circuit.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/cameraman.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/lena.jpg"
    path2 = "/home/sergio/TCG/PruebasImagenes/church1.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/nino.jpg"
    
    #HISTOGRAMA
    # drawHistogram(path2)

    # FILTROS
    # media(path, 11)
    # media_ponderada(path, 10)
    # mediana(path)
    # maxmin_max(path)
    # maxmin_max(path)
    # gauss(path)

    #BORDES
    # derivada1(path2)
    # derivada2(path2)

    laPlace(path2)
    # roberts(path2)
    # sobel(path2)
    # prewitt(path2)



