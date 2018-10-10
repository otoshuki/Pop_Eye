#Kriti 2018

#Import required libraries
import cv2
import numpy as np
import serial
import time
from calibrate import border
from scipy.optimize import *
import math
import struct

#Set up Serial Port
#arduino = serial.Serial('/dev/ttyACM0', 9600)

#frame = cv2.imread('Check1.jpg')
#new = frame.copy()
#Global var
centre_x = 0
centre_y = 0
#Arm Head
arm_x = 0
arm_y = 0
mapper = 0
popped = 0
#cv2.circle(frame,(400,225),2,(0,0,0),2)
##############Functions for Control##################

#Detects only one though
#Detect particular colored balloons
def detect(color,frame):
    detected = 0
    flag = 0
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
        lower = np.array([100,70,80])
    elif color == 'G':
        upper = np.array([80,255,255])
        lower = np.array([50,60,60])
    elif color == 'Y':
        upper = np.array([40,255,255])
        lower = np.array([20,100,100])
    #Loop through instructions
    while detected < 50:
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
        mask = cv2.GaussianBlur(mask,(5,5),2)
        #Apply Hough Circle Transform to get the circles
        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
        param1 = 100,param2 = 20,minRadius = 35,maxRadius = 60)
        #Convert to numoy array
        circles = np.uint16(np.around(circles))
        #Get the circles and draw them
        for i in circles[0,:]:
            cv2.circle(new,(i[0],i[1]),i[2],(255,255,255),2)
            #Append the circles to out
            out.append([i[0],i[1],i[2]])
            cv2.circle(new,(i[0],i[1]),2,(0,0,0),5)
        #cv2.circle(new,(100,100),35,(255,255,255),2)
        detected += 1
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

#Find distance between two points
def dist(x1,y1,x2,y2):
    distance = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    distance = np.sqrt(distance)
    return (distance*mapper)

#Find angle between two points and origin using dot product
def angle(ox,oy,x1,y1,x2,y2):
    C1 = np.array([x1-ox,y1-oy])
    C2 = np.array([x2-ox,y2-oy])
    dot = C1.dot(C2)
    mag = np.sqrt(C1.dot(C1)*C2.dot(C2))
    ang = np.arccos(dot/mag)*180/np.pi
    if x1 > ox:
        ang = -1*ang
    return ang

#Go detection seq
def seq(color):
    #Got though twice for check
    for iter in range(2):
        try:
            cap = cv2.VideoCapture(1)
            for i in range(50):
                ret,frame = cap.read()
            cap.release()
            #frame = cv2.imread('Check1.jpg')
            frame = calib(frame,0)
            pos = detect(color,frame)
            print('Number of ' + color + ' balloons : ' + str(len(pos)))
        except:
            print(color + ' Balloons Popped')
            break
        #Go through the seq for the number of balloons
        for i in range(len(pos)):
            #Find distances
            distance = int(dist(pos[i][0],pos[i][1],centre_x,centre_y))
            #Find angles
            ang = int(angle(centre_x,centre_y,pos[i][0],pos[i][1],
            arm_x, arm_y))
            #Move the arm to the position
            ret = move(frame,distance,ang,color)
            if ret == 0:
                break
            #Wait
            #Go for the next balloon

#Simulation Algorithm to move arm to required coordinates
def move1(frame,r,theta,color,x,y):
    global popped
    #Detect if the balloons are actually present
    try:
        pos = detect(color,new)
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
        cv2.circle(new,(x,y),70,(0,0,0),-1)
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
        #cv2.circle(frame,(int(x1),int(y1)),50,(0,0,0),-1)
        #print('Opposite Balloon was popped')
        popped += 1
        return 1
    else:
        print('All the colors already popped')
        return 0

#Algorithm to move arm to required coordinates
def move(frame,r,theta,color):
    global popped
    h = -10
    #Detect if the balloons are actually present
    try:
        pos = detect(color, frame)
        #balloons left
        left = len(pos)
    #If error returned, no balloons left
    except:
        left = 0
        print('Left = 0')
        return 0
    if left != 0:
        def FUN(z):
        	x=z[0]
        	y=z[1]
        	F = np.empty((2))
        	F[0] = 25.3*math.cos(x)+25.5*math.cos(y)-r
        	F[1] = 25.3*math.sin(x)+25.5*math.sin(y)-h
        	return F
        #Send the serial data as r,theta to arduino
        ZG= np.array([0.1,0.1])
        Z = fsolve(FUN,ZG)
        Z = Z* 180 / math.pi
        angle1 = (int(Z[0]))
        angle2 = (int(180-(Z[0]-Z[1])))
        direction = 0
        if theta < 0:
            direction = 1
        arduino.write(struct.pack('>BBBB', angle1,angle2,abs(theta),direction))
        print(struct.pack('>BBBB', angle1,angle2,abs(theta),direction))
        #Wait
        #Wait for confirmation
        #while arduino.readline() != 1:
        time.sleep(7)
        print('Balloon was popped')
        #Move to (r,theta+180)
        if theta < 0:
            theta += 180
        elif theta > 0:
            theta -= 180
        if theta < 0:
            direction = 1
        arduino.write(struct.pack('>BBBB', angle1,angle2,abs(theta),direction))
        print(struct.pack('>BBBB', angle1,angle2,abs(theta),direction))

        #Wait
        #Wait for confirmation
        #while arduino.readline() != 2:
        time.sleep(7)
        print('Opposite Balloon was popped')
        popped += 1
        return 1
    else:
        print('All the colors already popped')
        return 0

#Calibration
def calib(img,flag):
    #calibration
    global centre_x
    global centre_y
    global arm_x
    global arm_y
    print('Starting Calibration')
    crop = border(img)
    img = img[crop[0]:crop[1],crop[2]:crop[3]]
    #Set mapping variable
    global mapper
    mapper = 100/(crop[3]-crop[2])
    #Frame Centre
    centre_y,centre_x,channels = img.shape
    centre_x = int(centre_x/2)
    centre_y = int(centre_y/2)
    #Arm Head
    arm_x = centre_x
    arm_y = 0
    print('Calibrated')
    if flag == 1:
        print('Mapper = ' + str(mapper))
        print('cx = ' + str(centre_x) + ' cy = ' + str(centre_y))
        print('arm_x = ' + str(arm_x) + ' arm_y = ' + str(arm_y))
    return img

def calib1(flag):
    #calibration
    global centre_x
    global centre_y
    global arm_x
    global arm_y
    print('Starting Calibration')
    cap = cv2.VideoCapture(1)
    while True:
        ret,img = cap.read()
        #crop = border(img)
        #img = frame[crop[0]:crop[1],crop[2]:crop[3]]
        #Set mapping variable
        #global mapper
        #mapper = 100/(crop[3]-crop[2])
        #Frame Centre
        centre_y,centre_x,channels = img.shape
        centre_x = int(centre_x/2)
        centre_y = int(centre_y/2)
        #Arm Head
        arm_x = centre_x
        arm_y = 0
        print('Calibrated')
        cv2.imshow('NEW', img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    if flag == 1:
        print('Mapper = ' + str(mapper))
        print('cx = ' + str(centre_x) + ' cy = ' + str(centre_y))
        print('arm_x = ' + str(arm_x) + ' arm_y = ' + str(arm_y))
    return img
#Round 1
#'t' to be specified before round
def round1(t):
    #take the square/find the square
    det = 0
    print('STARTING ROUND 1')
    #Take input from camera
    #Treshold
    cap = cv2.VideoCapture(0)
    lower1 = np.array([0,100,100])
    lower2 = np.array([160,100,100])
    upper1 = np.array([10,255,255])
    upper2 = np.array([179,255,255])
    while det  < 10:
        if det == 0:
            for tf in range(10):
                ret,frame = cap.read()
            new = calib(frame,1)
        #Convert to HSV format
        hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
        #Create mask
        mask1 = cv2.inRange(hsv,lower1,upper1)
        mask2 = cv2.inRange(hsv,lower2,upper2)
        mask = cv2.addWeighted(mask1,1.0,mask2,1.0,0.0)
        #Apply morphological transformations
        #Find Contours
        img,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,
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
        #print('Contour Found')
        (cx,cy),rad = cv2.minEnclosingCircle(req_contour)
        print('aewad')
        cv2.circle(new,(int(cx),int(cy)), int(rad),(255,255,255),2)
        cv2.circle(new,(centre_x,centre_y), 10,(255,255,255),2)
        cv2.circle(new,(arm_x,arm_y), 10,(255,255,255),2)
        #Show results
        cv2.imshow('MASK',mask)
        cv2.imshow('NEW',new)
        #r = dist(centre_x,centre_y,cx,cy)
        #theta = angle(centre_x,centre_y,cx,cy,arm_x,arm_y)
        #print('r : ' + str(r) + ' ; theta : ' + str(theta))
        det += 1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #Find (r,theta) for (cx,cy)
    r = dist(centre_x,centre_y,cx,cy)
    theta = angle(centre_x,centre_y,cx,cy,arm_x,arm_y)
    cap.release()
    cv2.destroyAllWindows()
    print('RED MARKER Detected')
    print('r : ' + str(r) + ' ; theta : ' + str(theta))
    #move(frame,r,theta,color)
    time.sleep(2)
    #Break
    print('Round over')

#Round 2
def round2():
    print('STARTING ROUND 2')
    for color in ['R','B','G','Y']:
        seq(color)

#Round 3 - Part 1
#Max 5 targets
#Start pos - Pop targets set in difficult cases
#targets at close angle, close rad, limiting rad
def round3_p1():
    print('STARTING ROUND 3 - PT. 1')
    for color in ['R','B','G','Y']:
        seq(color)

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
        if round == 'cal':
            calib1(1)
        elif round == '0':
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
            print(popped)
            print('CLOSING')
            break

#Run the program
if __name__ == '__main__':
    main()
