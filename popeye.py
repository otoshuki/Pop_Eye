#Kriti 2018

#Import required libraries
import cv2
import numpy as np
import serial
import time

#Arm Head
arm_x = 400
arm_y = 100
#Frame Centre
centre_x = 800/2
centre_y = 450/2
frame = cv2.imread('Base.png')
cv2.circle(frame,(400,225),2,(0,0,0),2)
##############Functions for Control##################

#Detects only one though
#Detect particular colored balloons
def detect(color):
    detected = 0
    out = []
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
        lower = np.array([110,150,150])
    elif color == 'G':
        upper = np.array([70,255,255])
        lower = np.array([50,150,100])
    elif color == 'Y':
        upper = np.array([40,255,255])
        lower = np.array([20,100,100])
    #Loop through instructions
    while detected < 10:
        #ret,frame = cap.read()
        #Convert to HSV format

        new = frame.copy()
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

        #Apply Hough Circle Transform to get the circles
        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
        param1 = 100,param2 = 12,minRadius = 20,maxRadius = 300)
        #Convert to numoy array
        circles = np.uint16(np.around(circles))
        #Get the circles and draw them
        for i in circles[0,:]:
            cv2.circle(new,(i[0],i[1]),i[2],(255,255,255),2)
            #Append the circles to out
            out.append([i[0],i[1],i[2]])
            cv2.circle(new,(i[0],i[1]),2,(0,0,0),5)
        #Show windows
        cv2.imshow('NEW',new)
        cv2.imshow('MASK',mask)
        #Wait for key to be pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #Release the camera input and close windows
    #cap.release()
    cv2.destroyAllWindows()
    #Set out
    out = np.unique(np.uint16(out),axis=0)
    #print(out)
    #Return the average values
    return (out)

#Arm head detection
def arm_detect():
    detect = 0
    #Take input from camera
    #cap = cv2.VideoCapture(0)
    #Ranges for head colors
    lower = np.array([135,100,100])
    upper = np.array([155,255,255])
    while detect < 5:
        #Get the camera input
        #ret, frame = cap.read()
        #frame = cv2.imread('Base.png')
        #Convert to HSV color model
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #Create the mask
        mask = cv2.inRange(hsv,lower,upper)
        #Morphological transformation
        #Find the circle for head
        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
        param1 = 100,param2 = 10,minRadius = 20,maxRadius = 300)
        #Convert to numoy array
        new = frame.copy()
        circles = np.uint16(np.around(circles))
        #Get the circles and draw them
        #for i in circles[0,:]:
        #    cv2.circle(new,(i[0],i[1]),i[2],(255,255,255),2)
        #    cv2.circle(new,(i[0],i[1]),2,(0,0,0),5)
        #cv2.imshow('NEW',new)
        #Find the x and y coordinates
        detect += 1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #cap.release()
    cv2.destroyAllWindows()
    x = circles[0][0][0]
    y = circles[0][0][1]
    return (x,y)

#Find distance between two points
def dist(x1,y1,x2,y2):
    distance = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    distance = np.sqrt(distance)
    return (distance)

#Find angle between two points and origin using dot product
def angle(ox,oy,x1,y1,x2,y2):
    C1 = np.array([x1-ox,y1-oy])
    C2 = np.array([x2-ox,y2-oy])
    dot = C1.dot(C2)
    mag = np.sqrt(C1.dot(C1)*C2.dot(C2))
    angle = np.arccos(dot/mag)*180/np.pi
    return int(angle)

#Go detection seq
def seq(color):
    #Got though twice for check
    for iter in range(2):
        try:
            #Detect the balloon positions
            pos = detect(color)
            print('Number of ' + color + ' balloons : ' + str(len(pos)))
        except:
            print('Popped')
            break
        #Go through the seq for the number of balloons
        for i in range(len(pos)):
            #Find distances
            distance = int(dist(pos[i][0],pos[i][1],centre_x,centre_y))
            #Find angles
            ang = int(angle(centre_x,centre_y,pos[i][0],pos[i][1],
            arm_x, arm_y))
            #Move the arm to the position
            ret = move(distance,ang,color,pos[i][0],pos[i][1])
            if ret == 0:
                break
            #Wait
            #Go for the next balloon
            print(i)

#Algorithm to move arm to required coordinates
def move(r,theta,color,x,y):
    #Detect if the balloons are actually present
    try:
        pos = detect(color)
        #balloons left
        left = len(pos)
    #If error returned, no balloons left
    except:
        left = 0
        return 0
    if left != 0:
        #Send the serial data as r,theta to arduino
        #Wait
        #Wait until the balloon is popped
        cv2.circle(frame,(x,y),50,(0,0,0),-1)
        print('Balloon was popped')
        #Send the data for opposite balloon
        #Wait
        #Wait until the balloon is popped
        x0 = 800/2
        y0 = 450/2
        x1 = 0
        y1 = 0
        if x > x0:
            x1 = x0 - (x-x0)
        else:
            x1 = x0 + (x0-x)
        if y > y0:
            y1 = y0 - (y-y0)
        else:
            y1 = y0 + (y0-y)
        cv2.circle(frame,(int(x1),int(y1)),50,(0,0,0),-1)
        print('Opposite Balloon was popped')
        return 1
    else:
        print('All the colors already popped')
        return 0


##############Functions for Specific Steps##################
#1 target = 2 balloons(diametrically opposite)
#Always start off from an initial 9x9 position

#Round 1
#Reach a 15x15 red squre
#Reach diametrically opposite point and stay for 't' seconds
#'t' to be specified before round
def round1(t):
    #take the square/find the square
    print('STARTING ROUND 1')
    #Take input from camera
    #Treshold
    cap = cv2.VideoCapture(0)
    lower1 = np.array([0,100,100])
    lower2 = np.array([160,100,100])
    upper1 = np.array([10,255,255])
    upper2 = np.array([179,255,255])
    while True:
        ret,frame = cap.read()
        #Convert to HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Create mask
        mask1 = cv2.inRange(hsv,lower1,upper1)
        mask2 = cv2.inRange(hsv,lower2,upper2)
        mask = cv2.addWeighted(mask1,1.0,mask2,1.0,0.0)
        #Apply morphological transformations
        #Find Contours
        img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
        #Find largest contour
        max_area = 50
        max_i = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > max_area):
                max_area = area
                max_i = i
        req_contour = contours[max_i]
        #Draw contour
        new = frame.copy()
        print('Contour Found')
        (cx,cy),rad = cv2.minEnclosingCircle(req_contour)
        cv2.circle(new,(int(cx),int(cy)), int(rad),(255,255,255),2)
        #Show results
        cv2.imshow('MASK',mask)
        cv2.imshow('NEW',new)
        #Find (r,theta) for (cx,cy)
        r = dist(arm_x,arm_y,cx,cy)
        theta = angle(centre_x,centre_y,cx,cy,arm_x,arm_y)
        #Move to (r,theta)
        #Wait for t seconds
        #Move to diametrically opposite point
        #Wait for t seconds
        #Come back to Intial Postion
        #Wait
        #Break
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #cap.release()
    cv2.destroyAllWindows()
#Round 2
#Choose no of targets(2-12)
#Choose accuracy i.e. no. of colors(RBGY)
#Pop the balloons in the same sequence(RBGY)
def round2():
    print('STARTING ROUND 2')
    for color in ['R','B','G','Y']:
        seq(color)
    print('not done yet')

#Round 3 - Part 1
#Max 5 targets
#Start pos - Pop targets set in difficult cases
#targets at close angle, close rad, limiting rad
def round3_p1():
    print('STARTING ROUND 3 - PT. 1')
    print('not done yet')


#Round 3 -Part 2
#Max 5 targets
#Start pos - pop - return to Start pos
#Within time limit
def round3_p2():
    print('STARTING ROUND 3 - PT. 2')
    print('not done yet')

#Round 4 - tie breaker - do Round 2 and 3 again
############################################################

#Main function
def main():
    #Intitilization
    print('Pop-Eye -- Kameng -- Kriti 2018')
    print('\n')
    #Open Serial port
    #ser = serial.Serial('/dev/ttyUSB0')
    #Check port used
    #print("Serial Used : " + ser.name)
    #Select round
    while True:
        round = input("Enter the round : ")
        if round == '0':
            print('Initializing')
            #Arm Initial Position
            arm_x = 400
            arm_y = 100
            #arm_x = input('Enter initial x : ')
            #arm_y = input('Enter initial y : ')
        elif round == '1':
            time_still = input('Enter t : ')
            round1(int(time_still))
        elif round == '2':
            round2()
        elif round == '3':
            round3_p1()
        elif round == '4':
            round3_p2()
        else:
            print('CLOSING')
            break

#Run the program
if __name__ == '__main__':
    main()
