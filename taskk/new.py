
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import imutils
import matplotlib.pyplot as plt

from colorsys import rgb_to_hsv


def remove(alist):
    new_list=[]
    for elem in alist:
        if elem not in new_list:
            new_list.append(elem)
    alist = new_list
    return alist

def right(alist):
    new=[]
    for elem in range(len(alist)):
        if alist[elem] not in new:
            new.append(alist[elem])
        else:
            
            if((elem)%2)==0:
               new.insert(new.index(alist[elem])+1,alist[elem+1])
    alist = new
    print(alist)
    return alist




######################################################################################



def shifft(indexx,alist):
    if indexx == 0:
        alist = alist[-1:] + alist[:-1]
        #alist[0],alist[1]=alist[2],alist[3]
    elif indexx == 2:
        alist = alist[1:] + alist[:1]
    elif indexx ==3:
        alist = alist[-2:] + alist[:-2]
    return alist
        
        # find the biggest contour and it's area


def pre_images(images):
    new_images = []
    (height, width) = images[1].shape[:2]

    for image in images:
        h, w = image.shape[:2]
        image = cv2.resize(image, (w, height))
        #print(image.shape)
        new_images.append(image)

    return new_images

def create(image):
        
        img = np.zeros(image.shape, np.uint8) 
        img = cv2.bitwise_not(img)
        #upper_list.append(img)

        return img 

