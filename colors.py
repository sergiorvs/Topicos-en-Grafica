import numpy as np
import cv2
import math 
import matplotlib.pyplot as plt


###############utils##################
def min(r ,g ,b):
    if(r<g and r<b): return r
    if(g<r and g<b): return g
    return b
def max(r ,g ,b):
    if(r>g and r>b): return r
    if(g>r and g>b): return g
    return b
######################################

def RGB(image_path):
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # rows, cols = img.shape[:2]
    r, g, b = cv2.split(img)
    zeros = np.zeros_like(b)

    # blue = img[]
    

    # blue = img.copy()
    # blue[:,:,2] = 0
    # blue[:,:,1] = 0
    cv2.imshow('Original', img)
    cv2.imshow('blue', cv2.merge( (b, zeros, zeros) ) )
    cv2.imshow('green', cv2.merge( (zeros, g, zeros) ) )
    cv2.imshow('red', cv2.merge( (zeros, zeros, r) ) )

    # for i in range(1, rows-1):
    #     for j in range(1, cols-1):


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CMY(image_path):
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    height,width,channel = img.shape
    img_cmy = np.zeros((height,width,3))

    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            c = 1 - r_
            m = 1 - g_
            y = 1 - b_

            img_cmy.itemset((i,j,0),int(c*100))
            img_cmy.itemset((i,j,1),int(m*100))
            img_cmy.itemset((i,j,2),int(y*100))

    # mn = min(c, m, y)
    # c = (c-mn) / (1-mn)
    # m = (m-mn) / (1-mn)
    # y = (y-mn) / (1-mn)

    cv2.imwrite('image_cmy.jpg', img_cmy)

    c,m,y = cv2.split(img_cmy)
    zeros = np.zeros_like(c, dtype = float)

    cv2.imshow('CMY', img_cmy )
    # cv2.imshow('C', cv2.merge( (c, zeros, zeros) ) )
    # cv2.imshow('M', cv2.merge( (zeros, m, zeros) ) )
    # cv2.imshow('Y', cv2.merge( (zeros, zeros, y) ) )
    cv2.imshow('C', c)
    cv2.imshow('M', m)
    cv2.imshow('Y', y)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSI(image_path):
    img = cv2.imread(image_path)
    height,width,channel = img.shape
    img_hsi = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Cmin = min(r_,g_,b_)
            sum = r_ + g_ + b_

            #theta
            num = (1/2.)*( (r_ - g_) + (r_ - b_) )  # numerador
            dem = math.sqrt( (r_ - g_)**2 + (r_ - b_)*(g_ - b_) )  # denominador
            if(dem!=0): div = num/dem
            else: div = 0
            theta = math.acos(div)
            #Hue
            H = theta
            if(b_ > g_): H = 360-theta 
            #Saturation
            S = 1-(3/sum)*Cmin
            #Intensity
            I = sum/3

            img_hsi.itemset((i,j,0),int(H*255))
            img_hsi.itemset((i,j,1),int(S*255))
            img_hsi.itemset((i,j,2),int(I*255))
    
    cv2.imwrite('image_hsi.jpg', img_hsi)
    h,s,i = cv2.split(img_hsi)
    zeros = np.zeros_like(h, dtype = float)

    cv2.imshow('HSI', img_hsi)
    cv2.imshow('H', h )
    cv2.imshow('S', cv2.merge( (zeros, s, zeros) ) )
    cv2.imshow('I', cv2.merge( (zeros, zeros, i) ) )

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSV(image_path):
    img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_hsv = np.zeros((height,width,3))
    

    # CALCULATE
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.
            Cmax = max(r_,g_,b_)
            Cmin = min(r_,g_,b_)
            delta = Cmax-Cmin

            # Hue Calculation
            if delta == 0:
                H = 0
            elif Cmax == r_ :
                H = 60 * (((g_ - b_)/delta) % 6)
            elif Cmax == g_:
                H = 60 * (((b_ - r_)/delta) + 2)
            elif Cmax == b_:
                H = 60 * (((r_ - g_)/delta) + 4)

            # Saturation Calculation
            if Cmax == 0:
                S = 0
            else :
                S = delta / Cmax
            
            # Value Calculation
            V = Cmax 
            
            # Set H,S,and V to image
            img_hsv.itemset((i,j,0),int(H))
            img_hsv.itemset((i,j,1),int(S))
            img_hsv.itemset((i,j,2),int(V))

    # Write image
    cv2.imwrite('image_hsv.jpg', img_hsv)

    h,s,v = cv2.split(img_hsv)
    zeros = np.zeros_like(h, dtype = float)
    # View image
    cv2.imshow('HSV', img_hsv)
    cv2.imshow('H', h )
    cv2.imshow('S', cv2.merge( (zeros, s, zeros) ) )
    cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def YUV(image_path):
    img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_yuv = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            U = -0.147*r_ -0.289*g_ + 0.436*b_
            V = 0.615*r_ - 0.515*g_ - 0.100*b_

            img_yuv.itemset((i,j,0),int(Y*255))
            img_yuv.itemset((i,j,1),int(U*255))
            img_yuv.itemset((i,j,2),int(V*255))
    
    cv2.imwrite('image_yuv.jpg', img_yuv)

    y,u,v = cv2.split(img_yuv)
    zeros = np.zeros_like(y, dtype = float)

    cv2.imshow('YUV', img_yuv)
    cv2.imshow('Y', y )
    # cv2.imshow('U', cv2.merge( (zeros, u, zeros) ) )
    # cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )
    cv2.imshow('U', u )
    cv2.imshow('V', v )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def YIQ(image_path):
    img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_yiq = np.zeros((height,width,3))

    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            I = 0.596*r_ - 0.275*g_ - 0.321*b_
            Q = 0.212*r_ - 0.523*g_ + 0.311*b_

            img_yiq.itemset((i,j,0),int(Y*255))
            img_yiq.itemset((i,j,1),int(I*255))
            img_yiq.itemset((i,j,2),int(Q*255))
    
    # Write image
    cv2.imwrite('image_yiq.jpg', img_yiq)

    y,i,q = cv2.split(img_yiq)
    zeros = np.zeros_like(y, dtype = float)
    #View image
    cv2.imshow('YIQ', img_yiq)
    cv2.imshow('Y', y )
    cv2.imshow('I', cv2.merge( (zeros, i, zeros) ) )
    cv2.imshow('Q', cv2.merge( (zeros, zeros, q) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def YCrCb(image_path):
    img = cv2.imread(image_path)
    
    height,width,channel = img.shape
    img_ycrcb = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            R = img.item(i,j,0)
            G = img.item(i,j,1)
            B = img.item(i,j,2)

            Y =   16 +  65.738*R/256. + 129.057*G/256. +  25.064*B/256.
            Cb = 128 -  37.945*R/256. -  74.494*G/256. + 112.439*B/256.
            Cr = 128 + 112.439*R/256. -  94.154*G/256. -  18.285*B/256.

            # print(Y, Cb, Cr)
            
            img_ycrcb.itemset((i,j,0),int(Y))
            img_ycrcb.itemset((i,j,1),int(Cr))
            img_ycrcb.itemset((i,j,2),int(Cb))



    # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 



    y,cr,cb = cv2.split(img_ycrcb)
    zeros = np.zeros_like(y, dtype = float)

    # Write image
    cv2.imwrite('image_ycrcb.jpg', img_ycrcb)

    cv2.imshow('YCrCb', img_ycrcb)
    cv2.imshow('Y', y )
    cv2.imshow('Cr', cr)
    cv2.imshow('Cb', cb )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
#####FILTROS
def horizontalLine(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i-1][j-1])*-1 + int(img[i-1][j])*-1 + int(img[i-1][j+1])*-1 + int(img[i][j-1])*2 + int(img[i][j])*2 + int(img[i][j+1])*2 + int(img[i+1][j-1])*-1 + int(img[i+1][j])*-1 + int(img[i+1][j+1])*-1
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    
    cv2.imshow('horizontal lines', new_image)
    cv2.waitKey()  
    return new_image

def Line45(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i-1][j-1])*-1 + int(img[i-1][j])*-1 + int(img[i-1][j+1])*2 + int(img[i][j-1])*-1 + int(img[i][j])*2 + int(img[i][j+1])*-1 + int(img[i+1][j-1])*2 + int(img[i+1][j])*-1 + int(img[i+1][j+1])*-1
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    

    cv2.imshow('45 lines', new_image)
    cv2.waitKey()  
    return new_image

def verticalLine(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i-1][j-1])*-1 + int(img[i-1][j])*2 + int(img[i-1][j+1])*-1 + int(img[i][j-1])*-1 + int(img[i][j])*2 + int(img[i][j+1])*-1 + int(img[i+1][j-1])*-1 + int(img[i+1][j])*2 + int(img[i+1][j+1])*-1
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    

    cv2.imshow('vertical lines', new_image)
    cv2.waitKey()  
    return new_image

def Line_45(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i-1][j-1])*2 + int(img[i-1][j])*-1 + int(img[i-1][j+1])*-1 + int(img[i][j-1])*-1 + int(img[i][j])*2 + int(img[i][j+1])*-1 + int(img[i+1][j-1])*-1 + int(img[i+1][j])*-1 + int(img[i+1][j+1])*2
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    
    cv2.imshow('Line -45 degrees', new_image)
    cv2.waitKey()  
    return new_image

def points(image_path):
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape[:2]
    new_image = img.copy()

    cv2.imshow('Original', img)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum = int(img[i-1][j-1])*-1 + int(img[i-1][j])*-1 + int(img[i-1][j+1])*-1 + int(img[i][j-1])*-1 + int(img[i][j])*8 + int(img[i][j+1])*-1 + int(img[i+1][j-1])*-1 + int(img[i+1][j])*-1 + int(img[i+1][j+1])*-1
            # print(new_image[i][j])
            if(sum > 255):
                new_image[i][j] = 255
            elif(sum < 0):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum
    
    cv2.imshow('Points', new_image)
    cv2.waitKey() 

def convinepoints(image_path):
    new_image = horizontalLine(image_path) + Line45(image_path) + verticalLine(image_path) + Line_45(image_path)
    cv2.imshow('suma', new_image)
    cv2.waitKey() 


if __name__ == "__main__":
    path = "/home/sergio/TCG/imagenesTCG/lena.jpg"
    # path = "/home/sergio/TCG/PruebasImagenes/a1.jpg"
    # path = "/home/sergio/TCG/PruebasImagenes/lenaS.png"
    # path = "/home/sergio/TCG/PruebasImagenes/techo.png"
    # path = "/home/sergio/TCG/imagenesTCG/coins.png"
    # path = "/home/sergio/TCG/PruebasImagenes/circuit.jpg"
    # img = cv2.imread(path)

    # print(img[0])
    RGB(path)
    CMY(path)
    HSI(path)
    HSV(path)
    YUV(path)
    YIQ(path)
    YCrCb(path)

    # horizontalLine(path)
    # Line45(path)
    # verticalLine(path)
    # Line_45(path)
    # points(path)
    # convinepoints(path)
