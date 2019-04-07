import numpy as np
import cv2
from PIL import Image
import math 



def rotate(image_path, degrees_to_rotate, save_location):
    img_obj = Image.open(image_path)
    rotated_image = img_obj.rotate(degrees_to_rotate)
    rotated_image.save(save_location)
    rotated_image.show()

def invert_color(image_path, save_location):
    img_obj = Image.open(image_path)
    img_obj.show()
    img_matrix = np.array(img_obj)
    img_matrix = 255 - img_matrix
    img = Image.fromarray(img_matrix,'RGB')
    img.save(save_location)
    img.show()
    return img

def exp(image_path, integer_to_exp, save_location):
    img_obj = Image.open(image_path)
    img_obj.show()
    img_matrix = np.array(img_obj)
    img_matrix = np.exp(img_matrix)
    # for i in range (1, integer_to_exp):
    #     img_matrix = img_matrix * img_matrix
    img = Image.fromarray(img_matrix,'RGB')
    img.save(save_location)
    img.show()
    return img

def log(image_path, integer_to_exp, save_location):
    img_obj = Image.open(image_path)
    img_obj.show()
    img_matrix = np.array(img_obj)
    img_matrix = np.log(img_matrix)
    img = Image.fromarray(img_matrix,'RGB')
    img.save(save_location)
    img.show()
    return img

def move(image_path, tx, ty):
    img = cv2.imread(path)
    num_rows, num_cols, chanel = img.shape[:3]  # get the size of the matrix
    translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])

    img_translation = img.copy()

    for i in range (0, num_rows):
        for j in range (0, num_cols):
            img_translation[i][j] = 0

    cont_rows = 0
    for i in range (ty, num_rows):
        cont_cols = 0
        for j in range (tx, num_cols):
            img_translation[i][j][0] = img[cont_rows][cont_cols][0]
            img_translation[i][j][1] = img[cont_rows][cont_cols][1]
            img_translation[i][j][2] = img[cont_rows][cont_cols][2]
            cont_cols += 1
        cont_rows += 1

    # img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows)) #traslete function with opencv
    cv2.imshow('Translation', img_translation)
    cv2.waitKey()


    

def scale(image_path, sx, sy):
    img = cv2.imread(path)
    num_rows, num_cols = img.shape[:2]
    cx, cy = np.indices((num_rows, num_cols))
    translation_matrix = np.float32([ [sx,0,0], [0,sy,0] ])
    img_matrix = np.array(img)

    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    cv2.imshow('Scale', img_translation)
    cv2.waitKey()


def crotate(image_path, alpha):
    img = cv2.imread(path)
    num_rows, num_cols = img.shape[:2]
    r_alpha = alpha*math.pi/180
    
    # cy, cx = np.indices((num_rows, num_cols))
    aux_x = num_cols/2.0
    aux_y = num_rows/2.0

    img_rotated = img.copy()

    for i in range (0, num_rows):
        aux_i = float(i)/(num_rows-1)
        if(i < aux_y):
            aux_i -= 1
        for j in range (0, num_cols):
            aux_j = float(j)/(num_cols-1)
            if(j < aux_x):
                aux_j -= 1
            dx = aux_i*math.cos(r_alpha) - aux_j*math.sin(r_alpha)
            dy = aux_i*math.sin(r_alpha) + aux_j*math.cos(r_alpha)
            # print("dx, dy", dx, dy)
            if(dy < 0):
                dy = (dy+1) * (num_rows-1)
            else:
                dy = dy * (num_rows-1)
            if(dx < 0):
                dx = (dx+1) * (num_cols-1)
            else:
                dx = dx * (num_cols-1)
            
            if int(dx) >= num_cols or int(dy) >= num_rows:
                img_rotated[i][j] = 0
            else:
                img_rotated[i][j] = img[int(dx)][int(dy)]
    
    cv2.imshow('Rotate', img_rotated)
    cv2.waitKey()

    # center = (num_cols/2, num_rows/2)
    # translation_matrix = cv2.getRotationMatrix2D(center, alpha, 1.0)
    # img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    # cv2.imshow('Rotate2', img_translation)
    # cv2.waitKey()
    


if __name__ == "__main__":
    path = "/home/sergio/TCG/imagenesTCG/cameraman.jpg"
    save = "/home/sergio/TCG/results/result.jpg"
    # invert_color(path, save)
    exp(path, 2 , save)
    # log(path, 2 , save)
    # crotate(path, 90)
    # move(path, 100, 80)
    # scale(path, 1, 0.5)



