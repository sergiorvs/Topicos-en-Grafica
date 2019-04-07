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

def media(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_image[i][j] = (int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + 
                               int(img[i-1][j]) + int(img[i][j]) + int(img[i+1][j]) + 
                               int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/9
    
    cv2.imshow('Media', new_image)
    cv2.waitKey()

def mediana(image_path):
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
            new_image[i][j] = int(img[i][j+1]) + int(img[i][j-1]) + int(img[i+1][j]) + int(img[i-1][j]) - 4*int(img[i][j])
    
    cv2.imshow('laPlace', new_image)
    cv2.waitKey()     


if __name__ == "__main__":
    path = "/home/sergio/TCG/PruebasImagenes/lenaG.png"
    path2 = "/home/sergio/TCG/PruebasImagenes/circuit.jpg"
    save = "/home/sergio/TCG/results/result.jpg"
    
    #HISTOGRAMA
    # drawHistogram(path2)

    # FILTROS
    # media(path)
    # mediana(path)
    # maxmin_max(path)
    # maxmin_min(path)
    # gauss(path)

    #BORDES
    # derivada1(path2)
    # derivada2(path2)
    laPlace(path2)



