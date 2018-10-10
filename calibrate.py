# KRITI - KAMENG 2018

#Import libraries
import cv2
import numpy as np

#cap = cv2.VideoCapture(0)
frame = cv2.imread('Check1.jpg')

#Function to detect black borders and crop image
def border(frame):
    det = 0
    #cap = cv2.VideoCapture(0)
    #threshold
    lower1 = np.array([0,0,0])
    upper1 = np.array([180,255,70])
    while det < 10:
        #ret,frame = cap.read()
        #Convert to HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Create mask
        mask = cv2.inRange(hsv,lower1,upper1)
        #Apply morphological transformations

        #Find Contours
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
        #Find largest contour
        borders = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 500:
                borders.append(i)
        #Get the borders
        #print(len(borders))
        #Get coordinates of contours
        new = frame.copy()
        #print('Contour Found')
        min_xy = 10000
        max_xy = 0
        min_coor = 0
        max_coor = 0
        #Get the max and min x,y
        for i in range(len(borders)):
            x,y,w,h = cv2.boundingRect(contours[borders[i]])
            xy = x*y
            if xy < min_xy:
                min_xy = xy
                min_coor = i
            if xy > max_xy:
                max_xy = xy
                max_xy = i
        #Draw Min
        x,y,w,h = cv2.boundingRect(contours[borders[min_coor]])
        cv2.rectangle(new,(x,y),(x+w,y+h),(255,255,255),3)
        min_x = x
        min_y = y
        #Draw Max
        x,y,w,h = cv2.boundingRect(contours[borders[max_coor]])
        cv2.rectangle(new,(x,y),(x+w,y+h),(255,255,255),3)
        max_x = x
        max_y = y
        #Crop
        crop = frame[min_y:max_y+h,min_x:max_x+w]
        #Show results
        #cv2.imshow('CROPPED',crop)
        #cv2.imshow('NEW',new)
        det += 1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #cap.release()
    cv2.destroyAllWindows()
    frame = frame[min_y:max_y+h,min_x:max_x+w]
    crop = [min_y,max_y+h,min_x,max_x+w]
    #print(crop)
    return crop

def color_d(color):
    global frame
    #Take input from camera
    #cap = cv2.VideoCapture(0)
    #Select the treshold according to color
    if color == 'R':
        lower1 = np.array([0,100,100])
        upper1 = np.array([10,255,255])
        lower2 = np.array([160,100,100])
        upper2 = np.array([179,255,255])
    elif color == 'B':
        upper = np.array([130,255,255])
        lower = np.array([100,70,80])
    elif color == 'G':
        upper = np.array([80,255,255])
        lower = np.array([50,60,60])
    elif color == 'Y':
        upper = np.array([40,255,255])
        lower = np.array([20,100,100])
    while True:
        #ret,frame = cap.read()
        #Convert to HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Filter out the colors out of range
        #Extra case for R --> two masks
        if color == 'R':
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)
        else:
            mask = cv2.inRange(hsv, lower, upper)
        #Apply morphological transformations for better accuracy

        #Show windows
        cv2.imshow('MASK',mask)
        #Wait for key to be pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #Release the camera input and close windows
    #cap.release()
    cv2.destroyAllWindows()
    #Set out
    #out = np.unique(np.uint16(out),axis=0)
    #print(out)
    #Return the average values
    #return (out)

def main():
    print('Finding Borders')
    border()
    print('Color Calibration')
    color_d('R')
    color_d('B')
    color_d('G')
    color_d('Y')

if __name__ == '__main__':
    main()
