import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import imutils
import matplotlib.pyplot as plt

def create_mask(image):
    width, height = image.shape[:2][::-1]
    roi = image[height//2 - 20: height//2 + 20, width//2 - 20: width//2 + 20]

    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
    upper = np.array(
        [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])

    image_to_thresh = image
    hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

    # kernel = np.ones((3, 3), np.uint8)
    # for red color we need to masks.
    mask = cv2.inRange(hsv, lower, upper)
    ##mask = cv2.erode(mask, (3, 3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #not_mask = cv2.bitwise_not(mask)
    # cv2.imshow("Mask", not_mask)

    # cv2.imshow('masked image' , cv2.bitwise_not(image , image , mask=not_mask))

    # cv2.imshow('roi', roi)
    # cv2.waitKey(0)

    return mask



def quantize_images(image):
    # image = cv2.imread('s1.png')
    (h, w) = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters=4)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    return quant

def preProcessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(closing, 0, 400)
    cv2.imshow('threshold', canny)

    return threshold

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        # if area > 500:
        length = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * length, True)

        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area

    return biggest, max_area


# order points for warp prespective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def prepare_images(original):

    image = original.copy()
    width, height = image.shape[:2][::-1]

    kernel = np.ones((3, 3), np.uint8)
    mask= create_mask(quantize_images(image))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # dilated = cv2.dilate(canny, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours  , heirarchy = cv2.findContours(canny , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    # print('number of contours :' , len(contours))

    biggest, max_area = biggestContour(contours)
    #print(max_area, biggest)
    cv2.drawContours(original, contours, -1, (0, 255, 0), 1)
    #cv2.imshow('masked image', original)
    #cv2.imshow('masked ', mask)
    cv2.waitKey(0)

    if biggest.size != 0:
        biggest = reorder(biggest)
        #cv2.drawContours(image , [biggest] , -1 , (0,255,0) , 3)
        #cv2.imshow('image' , image)
        # cv2.waitKey(0)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        cropped = cv2.warpPerspective(original, matrix, (width, height))

        return cropped

def pre_images(images):
    new_images = []
    (height, width) = images[1].shape[:2]

    for image in images:
        h, w = image.shape[:2]
        image = cv2.resize(image, (w, height))
        #print(image.shape)
        new_images.append(image)

    return new_images