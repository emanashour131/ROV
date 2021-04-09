import numpy as np
import cv2

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

def shifft(indexx,alist):
    if indexx == 0:
        alist = alist[-1:] + alist[:-1]
        #alist[0],alist[1]=alist[2],alist[3]
    elif indexx == 2:
        alist = alist[1:] + alist[:1]
    elif indexx ==3:
        alist = alist[-2:] + alist[:-2]
    return alist
        

def create(image):
        
        img = np.zeros(image.shape, np.uint8) 
        img = cv2.bitwise_not(img)
        #upper_list.append(img)

        return img 

