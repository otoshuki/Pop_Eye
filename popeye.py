#Import required libraries
import cv2
import numpy as nn
##############Functions for Control##################

#Detect particular colored balloons
def detect():
    #argument - color
    print('not done yet')
    #returns coordinates [x,y]

#Find distance between two points
def dist(x1,y1,x2,y2):
    distance = (x1-x2)(x1-x2)+(y1-y2)(y1-y2)
    distance = nn.sqrt(distance)
    return (distance)

#Find angle between two points and origin
def ang(o1,o2,x1,y1,x2,y2):
    #argunments - origin, C1, C2
    print('not done yet')

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
    #For manual entry of coordinates for debugging
    #print('Enter the coordinates as x1 y1 x2 y2')
    #C = [int(x) for x in input().split()]



#Run the program
if __name__ == '__main__':
    main()
