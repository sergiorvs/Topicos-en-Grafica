
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
    path = "/home/sergio/TCG/PruebasImagenes/lenaS.png"
    # path2 = "/home/sergio/TCG/PruebasImagenes/circuit.jpg"
    path2 = "/home/sergio/TCG/imagenesTCG/coins.png"
    # path2 = "/home/sergio/TCG/imagenesTCG/nino.jpg"
    # save = "/home/sergio/TCG/results/result.jpg"
 #BORDES
    # derivada1(path2)
    # derivada2(path2)

    laPlace(path2)
    # roberts(path2)
    # sobel(path2)
    # prewitt(path2)
