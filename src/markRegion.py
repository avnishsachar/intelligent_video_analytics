## To set virtual line and Region of Interest ##

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:28:17 2019

@author: Asus
"""
import cv2
import numpy as np
import tkinter.messagebox 
import math

def selectPoint (event,x,y,flags,params):
    global point_selected,point,old_point,count,pts,param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        param[1]+=1
        if param[1] <= 3:
            point = (x,y)
            param[2].append(point)
            if len(param[2]) == 2:
                cv2.line(param[0],pts[0],pts[1],(0,255,255),2)  
                cv2.imshow("LinePoints",param[0])
            if len(param[2]) == 3:
                cv2.circle(param[0],(x,y),4,(255,0,0),-1)
                cv2.imshow("Direction",param[0])

def markRoi(video):
    global param,pts
    
    case = 0
    cap = (cv2.VideoCapture(video))
    #cap = VideoStream(video).start()

    try:

        ret,image = cap.read()
        
        image = cv2.resize(image,(640,480))
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        
        count = 0
        pts = []
        param = [image,count,pts]        
        cv2.namedWindow("LinePoints")
        cv2.setMouseCallback("LinePoints",selectPoint)    
        cv2.imshow("LinePoints",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.namedWindow("Direction")
        cv2.setMouseCallback("Direction",selectPoint)    
        cv2.imshow("Direction",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        region = cv2.selectROI(image)
        cv2.destroyAllWindows()
        
        try:
            x1 , x2 , x3 = pts[0][0]-region[0] , pts[1][0]-region[0] , pts[2][0]-region[0]         #line end points wrt roi
            y1 , y2 , y3 = pts[0][1]-region[1] , pts[1][1]-region[1] , pts[2][1]-region[1]
            
            if x1 != x2 and y1 != y2:
                slope = float(y1-y2)/float(x1-x2)
                const = y1 - slope*x1              
                distSign = (slope*x3 + const - y3) / (np.sqrt(np.square(slope)+1))
                if y2 >y1:
                    angle = 180-np.degrees(math.atan2((y2-y1),(x2-x1))) 
                else:
                    angle = -1*np.degrees(math.atan2((y2-y1),(x2-x1))) 
            elif x1 == x2 and y1!=y2:
                slope = 'null'
                angle = 90
                distSign = x3-x1
            elif y1 == y2 and x1 != x2:
                slope = 0
                angle = 0
                distSign = y1 - y3
            else:
                print("error,same point")
                #insert tkinter message"    
                
            if angle >=90 and distSign < 0:                                             #line on left side
                case = 1
                x1_crit , x2_crit = x1-30 , x2-30
                y1_crit , y2_crit = y1+30 , y2+30
                
            elif angle >=90 and distSign > 0:                                           #line on right side
                case = 2
                x1_crit , x2_crit = x1+30 , x2+30
                y1_crit , y2_crit = y1-30 , y2-30
                
            elif angle <90 and distSign < 0:                                            #line on right side
                case = 3
                x1_crit , x2_crit = x1+30 , x2+30
                y1_crit , y2_crit = y1+30 , y2+30
                 
            elif angle <90 and distSign > 0:                                            #line on left side
                case = 4
                x1_crit , x2_crit = x1-30 , x2-30
                y1_crit , y2_crit = y1-30 , y2-30
                 
            else:
                print('Error')
                #insert tkinter warning
            line1 = ((x1,y1),(x2,y2))
            line2 = ((x1_crit,y1_crit),(x2_crit,y2_crit))
            
            info = [line1,line2,region,case,slope]
            
        except:
            
            info = "null_pts"
    except:
        
        
        info = "null_rtsp"
        
    return info
