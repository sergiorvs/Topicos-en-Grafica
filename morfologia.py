import numpy as np
import cv2
import math 
import time

from PIL import Image


def erode(img, kernel):
    return cv2.erode(img,kernel)
def dilate(img, kernel):
    return cv2.dilate(img,kernel)

def cuadradoMask(n):#numero impar
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,n):
        l=[]
        for j in range(0,n):
            l.append(255)
        cuadrado.append(l)
    print (cuadrado)
    return cuadrado
def diamanteMask(n):#numero impar
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,int(n/2)+1):
        l=[]
        q=int(n/2-i)
        for j in range(0,q):
            print (j)
            l.append(0)
        for o in range(q,n-q):
            l.append(255)
        for k in range(0,q):
            l.append(0)
        cuadrado.append(l)
    for i in range (len(cuadrado)-1,-1,-1):
        cuadrado.append(cuadrado[i])

    print (cuadrado)
    return cuadrado
def cruzMasks(n):
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,n):
        l=[]
        q=int(n/2)
        if i == int(n/2) :
            for i in range (0,n):
                l.append(255)
        else:
            for j in range(0,q):
                l.append(0)
            l.append(255)
            for k in range(0,q):
                l.append(0)
        cuadrado.append(l)

    print (cuadrado)
    return cuadrado


def _erosion(img , kernel):
    kernel = kernel * 255
    rows, cols = img.shape[:2]
    new_image = np.zeros_like(img, np.uint8)
    krows, kcols = kernel.shape[:2]
    for i in range(0, rows-krows):
        for j in range(0, cols-kcols):
            cont = 0
            for x in range(0, krows):
                for y in range(0, kcols):
                    if(img[i+x][j+y] == kernel[x][y]):
                        cont+=1
            if(cont == krows*kcols):
                new_image[i][j] = kernel[krows/2][kcols/2]
    
    cv2.imwrite('/home/sergio/TCG/results/erode.jpg', new_image)
    cv2.imshow('Erosion', new_image)
    cv2.waitKey()
    return new_image

def _dilate(img, kernel, x, y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])
    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                if A[i+x][j+y]==255:
                    for k in range(0,Mh):
                        for l in range(0,Mw):
                            if M[k][l]==255:
                                if i+k<Ah and j+l<Aw:
                                    if NI[i+k][j+l]!=255:
                                        NI[i+k][j+l]=255
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    img1.show()
    print (img1)
    return img1


def erosion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])

    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                contA=0
                contM=0
                for k in range(0,Mh):
                    for l in range(0,Mw):
                        if M[k][l]==255:
                            contM+=1
                            if i+k<Ah and j+l<Aw:
                                if A[i+k][j+l]==255:
                                    contA+=1
                if contA==contM:
                    NI[i+x][j+y]=255

    img1 = Image.fromarray(NI)
    #img.save('my.png')
    img1.show()
    #print (NI)
    return img1

def dilatacion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])
    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                if A[i+x][j+y]==255:
                    for k in range(0,Mh):
                        for l in range(0,Mw):
                            if M[k][l]==255:
                                if i+k<Ah and j+l<Aw:
                                    if NI[i+k][j+l]!=255:
                                        NI[i+k][j+l]=255
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    img1.show(title='dilate', command='fim')
    #print (NI)
    return img1

def binarizar(img):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    print(A)
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]>200:#255/2:
                NI[i][j]=255
            else:
                NI[i][j]=0
    print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    img1.show()
    return img1

def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)
def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)

def _op(img, kernel, x, y):
    new_image = dilatacion(erosion(img, kernel, x, y), kernel, x, y)
    return new_image
def _cl(img, kernel, x, y):
    new_image = erosion(dilatacion(img, kernel, x, y), kernel, x, y)
    return new_image

if __name__ == "__main__":
    # path = "/home/sergio/TCG/imagenesTCG/coins.png"
    # path = "/home/sergio/TCG/PruebasImagenes/circuit.bmp"
    path = "/home/sergio/TCG/PruebasImagenes/j.jpeg"
    # path = "/home/sergio/TCG/PruebasImagenes/bw.jpg"
    # path = "/home/sergio/TCG/PruebasImagenes/world.png"


    nmask = 7
    
    # M = cuadradoMask(nmask)
    # M = diamanteMask(nmask)
    M = cruzMasks(nmask)

    # kernel = np.array(M)
    # kernel = np.ones((5,5),np.uint8)

    # img = cv2.imread(path, cv2.THRESH_BINARY)
    img = Image.open(path).convert('L')
    img.show(title="original")

    # img = binarizar(img)
    # cv2.imshow('Original',img)
    # img.show()
    # _erosion(img, M)
    


    erosion(img, M, nmask/2, nmask/2)
    # _dilate(img,M,nmask/2,nmask/2)
    # _op(img, M, nmask/2, nmask/2)
    # _cl(img, M, nmask/2, nmask/2)

    example = dilate(img,kernel) - erode(img,kernel)

    # erosion = erode(img,kernel)  
    # dilatation = dilate(img,kernel)
    # opening = opening(img, kernel)
    # closing = closing(img, kernel)

    # cv2.imshow('er',erosion)
    # cv2.imshow('dil',dilatation)
    # cv2.imshow('op',opening)
    # cv2.imshow('cl',closing)

    # dl.show()

    cv2.imshow('ex', example)
    cv2.waitKey()