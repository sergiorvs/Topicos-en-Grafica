import numpy as np
import cv2
import math 
import time

import matplotlib.pyplot as plt

from PIL import Image

#morfologia
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
def erode(img, kernel):
    return cv2.erode(img,kernel)
def dilate(img, kernel):
    return cv2.dilate(img,kernel)

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
def dilatacion(img, M, x, y):
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

def intersection(img1,img2):
    height,width = img2.shape
    new_image = np.zeros_like(img1)
    # print(new_image)
    for i in range(height): 
        for j in range(width): 
            if(img1[i][j]==img2[i][j]):
                new_image[i][j]=255
            else:
                new_image[i][j]=0
    return new_image

#preprocesing:
def blackFrameRemovalL(img):
    height,width,channel = img.shape
    print(height,width)
    cropped_image = np.copy(img)
    rowsDelete = []
    colsDelete = []

    for i in np.arange(height):
        L = 0
        cont = 0
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r#/255.
            g_ = g#/255.
            b_ = b#/255.

            Cmax = max(r_,g_,b_)
            Cmin = min(r_,g_,b_)
            L = (Cmax+Cmin)/2   # se calcula el componente L de HSL
            if(L < 20): cont += 1    # se considera como un pixel oscuro si el componente L es menor que 20
        # si mas de la mitad de pixeles de la fila son oscuros se borra la fila
        if(cont > width/2):
            rowsDelete.append(i)
    

    for i in np.arange(width):
        L = 0
        cont = 0
        for j in np.arange(height):
            r = img.item(j,i,0)
            g = img.item(j,i,1)
            b = img.item(j,i,2)

            r_ = r#/255.
            g_ = g#/255.
            b_ = b#/255.

            Cmax = max(r_,g_,b_)
            Cmin = min(r_,g_,b_)
            L = (Cmax+Cmin)/2   # se calcula el componente L de HSL
            if(L < 20): cont += 1    # se considera como un pixel oscuro si el componente L es menor que 20
        # si mas de la mitad de pixeles de la fila son oscuros se borra la fila
        if(cont > height/2):
            colsDelete.append(i)


    cropped_image = np.delete(cropped_image, rowsDelete, axis=0)
    cropped_image = np.delete(cropped_image, colsDelete, axis=1)
    cv2.imshow('blackFrameRemoval', cropped_image)
    # cv2.waitKey()
    return cropped_image

def topHatTransform(img):
    kernel = np.ones((11,11),np.uint8)
    # kernel = diamanteMask(9)
    
    ret, img = cv2.threshold(img, 126, 255, cv2.THRESH_BINARY)
    # cv2.imshow('01', img)
    new_image = closing(img,kernel) - img
    cv2.imshow('tophat', new_image)
    # cv2.waitKey()
    return new_image

def blackFrameRemoval(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop
def gaussianFilterx(img):
    G = [1,2,1,2,4,2,1,2,1]
    # G = [1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]
    # G = np.array(G, dtype = float)*(1./273.)
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
def gaussianFilter(img):
    return cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

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
    
    # cv2.imshow('laPlace', new_image)
    # cv2.waitKey() 
    return new_image



def preprocesing(img):
    
    img = blackFrameRemovalL(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', img)
    img = gaussianFilter(img)
    cv2.imshow('gauss', img)
    img = topHatTransform(img)
    kernel = np.ones((7,7),np.uint8)
    # cv2.imshow('asdf', 255-img)
    # cv2.waitKey()
    for i in range(2):
        img = intersection(dilate(img,kernel), 255-img)
    return img


#Segmentation
def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    # print(outx, outy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

def region_growing(img, seed):
    list = []
    out = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
        out[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                out[coord[0], coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        cv2.imshow("progress",out)
        cv2.waitKey(1)
    return out

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Seed: ' + str(x) + ', ' + str(y), img[y,x]
        clicks.append((y,x))



if __name__ == "__main__":
    # path = "melanomas.png"
    # path = "bfr.png"
    path = "example.png"
    # path = "melanoma.jpg" 
    # path = "Melanoma.jpg"
    # path = "IMD002.bmp"
    # path = "IMD161.bmp"
    img = cv2.imread(path)
    cv2.imshow('original',img)

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # img = 255-img

    img = preprocesing(img)
    cv2.imshow('preprocesing',img)
    cv2.waitKey()

    clicks = []
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', img)
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(img, seed)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()
    cv2.destroyAllWindows()
