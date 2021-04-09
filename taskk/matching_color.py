import cv2
import numpy as np

from colorsys import rgb_to_hsv

def convert(color,colors):
    colors = np.array(colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return smallest_distance 


def convertt(col,ref_list):
    colors = ref_list
    color_to_match = col
    
    ds ,k= (min((abs(rgb_to_hsv(*k)[0]-rgb_to_hsv(*color_to_match)[0]),k) for k in colors))
    #print(k)
    return k
