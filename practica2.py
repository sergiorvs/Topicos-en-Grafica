import numpy as np
import cv2
from PIL import Image
import math 
from matplotlib import pyplot as plt
import time


# HISTOGRAMA

def drawHistogram(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    size = rows*cols
    cv2.imshow('Original', img)
    cv2.waitKey()
    
    # print("tam = ", rows, ", ", cols)



    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist,bins = np.histogram(img,256,[0,256])
    # print("histograma", bins)

    # for i in range(hist):
    #     print(hist[i])

    plt.plot(hist, color = 'gray')
    plt.xlim(0,256)
    plt.show()

    # print(plt.bin())

    # print(hist)
    print("the max value is: ", hist[255][0])
    print("the second value is: ", hist[254][0])

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
    hist = cv2.calcHist([ecualized_image], [0], None, [256], [0, 256])
    plt.plot(hist, color = 'blue')
    plt.show()

    cv2.waitKey()
    return ecualized_image



# FILTROS

def media(img,mask):
    cv2.imshow('Original', img)
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            cont=0
            sum=0
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    sum+=int(img[x][y])
                    cont+=1
            new_image[i][j] = int(sum/cont)  #(int(img[i-1][j-1]) + int(img[i][j-1]) + int(img[i+1][j-1]) + int(img[i-1][j]) + int(img[i][j]) + int(img[i+1][j]) + int(img[i-1][j+1]) + int(img[i][j+1]) + int(img[i+1][j+1]))/9
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

# def mediana(image_path, tam):
#     img = cv2.imread(image_path, 0)
#     rows, cols = img.shape[:2]
#     new_image = img.copy()

#     cv2.imshow('Original', img)

#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             ds = []
#             for x in range(0, tam):
#                 for y in range(0, tam):
#                     ds.append(img[i+x][j+y])
#             # ds.append(img[i-1][j-1])
#             # ds.append(img[i][j-1])
#             # ds.append(img[i+1][j-1])
#             # ds.append(img[i-1][j])
#             # ds.append(img[i][j])
#             # ds.append(img[i+1][j])
#             # ds.append(img[i-1][j+1])
#             # ds.append(img[i][j+1])
#             # ds.append(img[i+1][j+1])

#             ds = sorted(ds)
#             # print(ds)
#             new_image[i][j] = ds[tam/2]
    
#     cv2.imshow('Mediana', new_image)
#     cv2.waitKey()

def mediana(img,mask):
    cv2.imshow('Original', img)
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[int((mask*mask)/2)]
    cv2.imshow('Mediana', new_image)
    cv2.waitKey()


def maxmin_max(image_path):
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[(mask*mask)-1]
    cv2.imshow('Max', new_image)
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

def laPlace(img):
    # img = cv2.imread(image_path, 0)
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

def gauss2(img):
    # G = [1,2,1,2,4,2,1,2,1]
    G = [1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]
    G = np.array(G, dtype = float)*(1./273.)
    print(G)
    tam=3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]
            sum1=sum1/(16)
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
   
    cv2.imshow('Gaussx', M)
    cv2.waitKey()
    return M

        
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
    path = "/home/sergio/TCG/PruebasImagenes/lenaG.png"

    # path2 = "/home/sergio/TCG/PruebasImagenes/circuit.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/cameraman.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/lena.jpg"
    # path2 = "/home/sergio/TCG/PruebasImagenes/f1.jpg"
    # path2 = "/home/sergio/TCG/PruebasImagenes/f2.jpg"
    # path2 = "/home/sergio/TCG/a4.jpg"
    # path2 = "/home/sergio/TCG/imagenesTCG/nino.jpg"
    

    img = cv2.imread(path, 0)
    #HISTOGRAMA
    # drawHistogram(path2)

    # FILTROS
    # media(img, 11)
    # media_ponderada(path, 10)
    # mediana(img, 11)
    # maxmin_max(path2)
    # maxmin_max(path)
    gauss(path)
    # gauss2(img)
    dst = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Smoothing",dst)
    cv2.waitKey(0) # waits until a key is pressed

    #BORDES
    # derivada1(path2)
    # derivada2(path2)

    # laPlace(path2)
    # roberts(path2)
    # sobel(path2)
    # prewitt(path2)


    # laPlace(drawHistogram(path2))



