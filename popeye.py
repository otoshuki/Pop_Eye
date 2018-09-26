#Import required libraries
import cv2
import numpy as np

#Take input from camera
cap = cv2.VideoCapture(0)
##############Functions for Control##################

#Detect particular colored balloons
def detect(color):
    detected = 0
    #Select the treshold according to color
    if color == 'R':
        lower1 = np.array([0,100,100])
        upper1 = np.array([10,255,255])
        lower2 = np.array([160,100,100])
        upper2 = np.array([179,255,255])
    elif color == 'B':
        upper = np.array([130,255,255])
        lower = np.array([110,100,100])
    elif color == 'G':
        upper = np.array([70,255,255])
        lower = np.array([50,100,100])
    elif color == 'Y':
        upper = np.array([40,255,255])
        lower = np.array([20,100,100])

    #Loop through instructions
    while detected < 50:
        ret,frame = cap.read()
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
        #Get the balloon using Hough transform
        mask = cv2.medianBlur(mask,3)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 2)
        #Find the balloon
        im,contours,heirarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        max_area = 100
        max_i = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area>max_area):
                max_area = max_area
                max_i = i
        req_contour = contours[max_i]
        new = frame.copy()
        cv2.drawContours(new, contours, max_i, (0,255,255), 3)

        #Centroid
        moments = cv2.moments(req_contour)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        cv2.circle(new,(cx,cy),100,(0,255,0),2)
        #Draw the circle
        #try:
        #    for i in balloon:
        #        cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        #        cv2.circle(frame,(i[0],i[1]),1,(0,0,255),3)
        #    print('Balloon Found')
        #        detected += 1
        #except:
        #    print('Not found')

        cv2.imshow('FRAME',frame)
        cv2.imshow('NEW', new)
        cv2.imshow('MASK',mask)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    out = [int(i) for i in balloon]
    #out = [x,y,r]
    return (out)

#Find distance between two points
def dist(x1,y1,x2,y2):
    distance = (x1-x2)(x1-x2)+(y1-y2)(y1-y2)
    distance = nn.sqrt(distance)
    return (distance)

#Find angle between two points and origin using dot product
def ang(ox,oy,x1,y1,x2,y2):
    C1 = np.array([x1-ox,y1-oy])
    C2 = np.array([x2-ox,y2-oy])
    dot = C1.dot(C2)
    mag = np.sqrt(C1.dot(C1)*C2.dot(C2))
    angle = np.arccos(dot/mag)*180/np.pi
    return int(angle)

#Go through balloons in sequence
def seq():
    #argument - colors available
    print('not done yet')

#Algorithm to move arm to required coordinates
def move():
    #arguments - coordinates x,y
    print('not done yet')

##############Functions for Specific Steps##################
#1 target = 2 balloons(diametrically opposite)

#Initial position
#Starts at a 9x9 black marker
def start():
    print('not done yet')

#Round 1
#Stay in a 15x15 square for 't' seconds
#Reach diametrically opposite point and stay for 't' seconds
#'t' to be specified before round
def round1():
    print('not done yet')

#Round 2
#Choose no of targets(2-12)
#Choose accuracy i.e. no. of colors(RBGY)
#Pop the balloons in the same sequence(RBGY)
def round2():
    print('not done yet')

#Round 3 - Part 1
#Max 5 targets
#Start pos - Pop targets set in difficult cases
#targets at close angle, close rad, limiting rad
def round3_p1():
    print('not done yet')

#Round 3 -Part 2
#Max 5 targets
#Start pos - pop - return to Start pos
#Within time limit
def round3_p2():
    print('not done yet')

#Round 4 - tie breaker - do Round 2 and 3 again
############################################################

#Main function
def main():
    print('Not completed yet')
    detect('R')


#Run the program
if __name__ == '__main__':
    main()
