import cv2
import numpy as np
from matching_color import * 
from process_image import *
from assignList import *

def extract(res):
    blue = masked_image[res[1],res[0]][0]
    green= masked_image[res[1],res[0]][1]
    red = masked_image[res[1],res[0]][2]
    strr= (str(red)+','+str(green)+','+str(blue))

    return red,green,blue


if __name__ == "__main__":
    
    top = cv2.imread('s5.png')
    s2 = cv2.imread('s2.png')
    s3 = cv2.imread('s3.png')
    s4 = cv2.imread('s4.png')
    s1 = cv2.imread('s1.png')
    
    image_list=[]
    ls= [s1,s3,top,s4,s2]
    ls = pre_images(ls)

    listofColor = []
    listofColor2 = []
    listofColorT = []
    listofColorB = []
    sortedlist=[]
    lower_list=[]
    upper_list=[]
    
    listoftupleT=[]
    uupper_list=[]

    for i in ls:
        cropped = prepare_images(i)
        image_list.append(cropped)
        #cv2.imshow("cropped1", cropped)
    #print(image_list)


    def resize(img):
        img = cv2.resize(img,(200,200))
        return img
    
    for image in image_list:
        #image = cv2.resize(image,(500,500))

        image = prepare_images(image)
        mask = cv2.bitwise_not(create_mask(quantize_images(image)))
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print (len(cnts))

        contours =[]
       
        for con in cnts:
            arc_len = cv2.arcLength( con, True )
            approx = cv2.approxPolyDP( con, 0.1 * arc_len, True )
            if cv2.contourArea( con ) > 1000 :
                contours.append(con)
            
        
        length_of_contours = len(contours)
        #print(len(contours))

        if length_of_contours == 3:
            lower_list.append(image)
            #cv2.imshow('i',image)
            for y in lower_list:
                listoftuple=[]
                for c in contours:
                    #if length_of_contours == 3:
                    #cv2.drawContours(y, [c],-1, (0,255,0),2)
                    #cv2.imshow("con", y)
                    M = cv2.moments(c)
                    center_x, center_y = (int(M['m10'] / (M['m00']+ 1e-5)), int(M['m01'] / (M['m00']+ 1e-5)))
                    listoftuple.append((center_x, center_y))

                res =max(listoftuple, key = lambda i : i[0])[0]
                res2 = min(listoftuple, key = lambda i : i[0])[0]
                resT = min(listoftuple, key = lambda i : i[1])[0]
                
                for a in range(len(listoftuple)):
                    if res == listoftuple[a][0]:
                        res= listoftuple[a]
                    if res2 == listoftuple[a][0]:
                        res2= listoftuple[a]
                    if resT == listoftuple[a][0]:
                        resT= listoftuple[a]
               
                listofColor.append(extract(res))
                listofColor2.append(extract(res2))
                listofColorT.append(extract(resT))

                listofColor=remove(listofColor)
                listofColor2=remove(listofColor2)
                listofColorT=remove(listofColorT)
  
                print(listofColor)
                print(listofColor2)
                print(listofColorT)
        else:
            upper_list.append(image)
            #cv2.imshow('i',image)
            for image in upper_list:
                for c in contours:
                    #if length_of_contours == 3:
                    #cv2.drawContours(image, [c],-1, (0,255,0),2)
                    #cv2.imshow("con", image)
                    M = cv2.moments(c)
                    center_x, center_y = (int(M['m10'] / (M['m00']+ 1e-5)), int(M['m01'] / (M['m00']+ 1e-5)))
                    listoftupleT.append((center_x, center_y))

                res =max(listoftupleT, key = lambda i : i[0])[0]
                res2 = min(listoftupleT, key = lambda i : i[0])[0]
                resT = min(listoftupleT, key = lambda i : i[1])[0]
                resB = max(listoftupleT, key = lambda i : i[1])[0]

                for a in range(len(listoftupleT)):
                    if resB == listoftupleT[a][0]:
                        resB= listoftupleT[a]

                listofColorB.append(extract(resB))

    for x in range(len(listofColor)):

        xx= convertt(listofColor[x],listofColor2)
        sortedlist.append(x)
        o = listofColor2.index(xx)
        print(o)
        sortedlist.append(o)
    print(sortedlist)
    final_list = []

    sortedlist = right(sortedlist)
         
    print(sortedlist)

    
    for y in sortedlist: 
        if y not in final_list: 
            final_list.append(y)
    print(final_list)
    resulttt= [lower_list[i] for i in final_list]

     
    for x in range(len(listofColorB)):
        xx= convertt(listofColorB[x],listofColorT)
        indexx =listofColorT.index(xx)
    print(final_list.index(indexx))
    indexx = final_list.index(indexx)
    resulttt = shifft(indexx,resulttt)
    for n in range(4):
        if n == 1:
            uupper_list.append(upper_list[0])
        else:
            uupper_list.append(create(resulttt[n]))

       
           
    out = np.hstack(resulttt)
    cv2.imshow('Output', out)
    
    outt= np.hstack(uupper_list)
    cv2.imshow('OutputU', outt)
    
    #finnnnal=[uupper_list, resulttt]
    #f = np.vstack(finnnnal)
    #cv2.imshow("ff",f)

    cv2.waitKey(0)
    cv2.destroyAllWindows()