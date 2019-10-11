
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:19:17 2019

@author: Asus
"""
import cv2
import numpy as np
import time
import threading
from datetime import datetime
import queue
from datetime import date
import os
import pandas as pd
from playsound import playsound
import sys


#from notifications import *

global totalCount, totalAttempts, liveCount
totalCount = 0
totalAttempts = 10
liveCount = 0

##threading sub class##

class LineCrossing(threading.Thread):
    def __init__(self, name, rtsp, line1, line2, region, case, slope, livestream, threshold,location):

        threading.Thread.__init__(self)
        self.NAME = name
        self.RTSP = rtsp
        self.LINE1 = line1
        self.LINE2 = line2
        self.REGION = region
        self.CASE = case
        self.SLOPE = slope
        self.LIVESTREAM = livestream
        self.THRESHOLD = threshold
        self.LOCATION = location     # to store position of livestream windows on screen
        
    def run(self):
        global totalCount,totalAttempts,liveCount
        detectPeople(self.NAME, self.RTSP, self.LINE1, self.LINE2, self.REGION, self.CASE, self.SLOPE, self.LIVESTREAM,
                     self.THRESHOLD,self.LOCATION)
        if self.LIVESTREAM.lower() == 'yes':
            liveCount+=1
        
        if liveCount > 2:
            print(self.LIVESTREAM,liveCount)
            
        #restart process if feedloss occurs
        t = time.localtime()
        currentTime = time.strftime("%H:%M:%S", t)
        count = 0   #variable to track no of attempts for restarting a thread
        while count < totalAttempts:
            
            if feedFlag == 1:

                count+=1
                totalCount+=1
                detectPeople(self.NAME, self.RTSP, self.LINE1, self.LINE2, self.REGION, self.CASE, self.SLOPE,
                             self.LIVESTREAM, self.THRESHOLD, self.LOCATION)
                    
        if count == 10:
            
            fileN = os.path.join(os.getcwd(),"LineCrossed",str(date.today()))
            #if not os.path.exists(fileN):
            #    f = open(fileN,'w+')
            with open ("./LineCrossing/"+str(date.today())+"//notWorking.txt",'a') as infile:
                infile.write("Camera %s not working at %s\n"%(self.NAME,str(currentTime)))   
        cv2.destroyAllWindows()

    def get_id(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

## thread subclass to start sound alarm ##
class Alarm_sound(threading.Thread):

    def __init__(self, alarm_cam):
        threading.Thread.__init__(self)
        self.NAME = alarm_cam

    def run(self):
    	for i in range(2):
        	playsound('beep-02.mp3')


## Function to create the date wise and camera name folders ##
def check_location(location):
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except:
            print('Unable to create directory')


## to check if event occurred or not##
def checkCase(d1, d2, case):
    if case == 1 or case == 3:
        if d1 < 0 and d2 > 0:
            tFlag = 1
        elif d1 < 0 and d2 < 0:
            tFlag = 2
        else:
            tFlag = 0

    if case == 2 or case == 4:
        if d1 > 0 and d2 < 0:
            tFlag = 1
        elif d1 > 0 and d2 > 0:
            tFlag = 2
        else:
            tFlag = 0

    return tFlag


## implementation of line crossing ##
def detectPeople(name, rtsplink, line1, line2, region, case, slope, livestream, threshold, location):
    global direc, feedFlag, liveStream_frame, liveStream_name, event_occured, eventStream_path, eventStream_frame, livestream_loc 
    frame_queue = queue.Queue(maxsize=0)
    
    # flags initialisation
    eventFlag = False                       # If event occurs
    liveStream_frame = None                 # camera frame if livestream is on
    liveStream_name = None                  # camera name if livestream is on
    event_occured = False
    feedFlag = 0                            # Flag to check feedloss
    tresFlag = 0                            # Flag to check trespassing
    countFrame = 0                          # Count to skip frames to reduce fps
    eventStream_frame = None                #event frame
    eventStream_path = None                 #event path

    check_location(os.path.join(os.getcwd(), "LineCrossing", str(datetime.now()).split(' ')[0], str(name)))
    #Input_vid = "filesrc location="+rtsplink+" ! tsdemux ! queue ! h264parse ! nvv4l2decoder!  nvvidconv ! video/x-raw, format=RGBA, height=480, width=640! videoconvert ! appsink"
    # Input_vid = 'filesrc location=3.ts ! tsdemux ! queue ! h264parse ! nvv4l2decoder!  nvvidconv ! video/x-raw, format=RGBA, height=480, width=640! videoconvert ! appsink'
    # Input_vid = "rtspsrc location="+rtsplink + "rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)RGBA, height=(int)480, width=(int)640 ! videoconvert ! appsink"
    # Input_vid = "rtspsrc location="+rtsplink+"! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
    #Input_vid = "rtspsrc location=" + rtsplink + " ! rtph264depay ! h264parse ! nvv4l2decoder !  nvvidconv ! video/x-raw, format=RGBA, height=480, width=640! videoconvert ! appsink"
    
    Input_vid = rtsplink
    try:
        cap = cv2.VideoCapture(Input_vid)
        ret,frame_org = cap.read()
        

    except Exception as e:
        print("error")
        raise e

    line1 = (int(line1[0].split(',')[0]), int(line1[0].split(',')[1]), int(line1[1].split(',')[0]), int(line1[1].split(',')[1]))
    line2 = (int(line2[0].split(',')[0]), int(line2[0].split(',')[1]), int(line2[1].split(',')[0]), int(line2[1].split(',')[1]))

    if slope != 'null':
        const1 = line1[1] - slope * line1[0]  # to find eq of lines (y = mx + const)
        const2 = line2[1] - slope * line2[0]
    
    backSub = cv2.createBackgroundSubtractorKNN(20,600,True)
    #backSub = cv2.bgsegm.createBackgroundSubtractorGMG(10,.8)  #initialize background subtraction
    #backSub.setVarThreshold(12)            # to ignore small movements in the frame
    #print(threshold)
    while True:
        ret,frame_org = cap.read()
        countFrame += 1
        check_location(os.path.join(os.getcwd(),"LineCrossing",str(datetime.now()).split(' ')[0],str(name)))
        if countFrame%2 == 0:  # to skip frames to reduce fps
            if ret == True:
                frame_org = cv2.resize(frame_org,(640,480))     # reduce frame size
                pts = np.array(region)

                rect = cv2.boundingRect(pts)
                x,y,w,h = rect
                cropped = frame_org.copy()
                
                mask = np.zeros(cropped.shape[:2], np.uint8)
                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                
                dst = cv2.bitwise_and(cropped, cropped, mask=mask)
                
                bg = np.ones_like(cropped, np.uint8)*255
                cv2.bitwise_not(bg,bg, mask=mask)
                frame = bg+ dst
                
                
                for i in range(len(region)-1):                  # mark ROI on all frames 
                    cv2.line(frame_org,(region[i][0],region[i][1]),(region[i+1][0],region[i+1][1]),(0,255,0),2)
                    cv2.line(frame_org,(region[len(region)-1][0],region[len(region)-1][1]),(region[0][0],region[0][1]),(0,255,0),2)

                cv2.line(frame_org, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 255), 2)   # mark the warning line
                cv2.line(frame_org, (line2[0], line2[1]), (line2[2], line2[3]), (0, 0, 255), 2)     # mark the critical line
                
                cv2.imwrite(os.path.join(os.getcwd(), "LineCrossing", name,'marking_%s.jpg' % name), frame_org)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
                h,s,v = cv2.split(hsv)
                #sv = cv2.merge(s,v)
                #s.fill(0)
                #hsv = cv2.merge([h, s, v])
                thresh = backSub.apply(v,learningRate=0.2)       # apply back sub
                se = np.ones((3,3), dtype='uint8')
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
                kernel = np.ones((3,3),np.uint8)

                thresh = cv2.erode(thresh,kernel,iterations=1)
                #cv2.imshow("threshnew",thresh)
                #cv2.waitKey(1)
                
                # find contours of detected objects
                if cv2.__version__[0] == '4':
                    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    (_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in cnts:
                    if cv2.contourArea(c) < threshold:# or cv2.contourArea(c) > 3000:
                        continue
                    (x1, y1, w1, h1) = cv2.boundingRect(c)
                    
                    lwrCtr = (int(x1 + w1 / 2), int(y1 + h1))
                    cv2.rectangle(frame_org, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 200), 2)        # rectangle around detected object
                    cv2.circle(frame_org, lwrCtr, 1, (255, 255, 255), 5)                    # detect lower center of the detected object
                    if slope != 'null':
                        dist = ((slope) * (lwrCtr[0]) - (lwrCtr[1]))                
                        d1 = (dist + (const1))                              # distance of the object from warning line
                        d2 = (dist + (const2))                              # distance of the object from critical line
                    else:
                        d1 = lwrCtr[0] - line1[0][0]
                        d2 = lwrCtr[0] - line2[0][0]

                    tresFlag = checkCase(d1, d2, case)

                if tresFlag == 1 :#and cv2.contourArea(c) >= threshold:
                    today = str(date.today())
                    frame_name = str(datetime.now().strftime("%H-%M-%S"))
                    #print("warning",cv2.contourArea(c))
                    cv2.putText(frame_org, "WARNING", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.imwrite(os.path.join(os.getcwd(), "LineCrossing", today, name,'Warning_alert_%s.jpg' % frame_name), frame_org)
                    tresFlag = 0
                    
                elif tresFlag == 2 :#and cv2.contourArea(c) >= threshold-100:
                    
                    cv2.putText(frame_org, "CRITICAL", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #print(cv2.contourArea(c))
                    # Save frame with time for critical event
                    frame_name = str(datetime.now().strftime("%H-%M-%S"))
                    today = str(date.today())
                    alert_message = 'WARNING !TRESPASS ACTIVITY HAS BEEN DETECTED ON CAMERA %s.' % name   #text for sms
                    #sendSMS(alert_message)
                    #email_alert(name, rtsplink, ' WARNING!! Tresspass Event ')     
                    if not os.path.exists(os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name)):
                        cv2.imwrite(os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name), frame_org)
                        image_path = os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name)
                        frame_queue.put(image_path)             # add image path to queue
                        eventFlag = True                        # raise event flag
                        tresFlag = 0                            # reset trespass flag

                else:
                    cv2.putText(frame_org, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if livestream.lower() == 'yes':
                    liveStream_name = name          # pass camera name for livestream
                    liveStream_frame = frame_org    # pass camera frame for livestream
                    livestream_loc = location       # pass window location for livestream

                if eventFlag == True and not frame_queue.empty():
                    
                    eventStream_path = frame_queue.get()        # extract image path
                    eventStream_frame = frame_org               # pass event frame
                    event_occured = True                        # raise event flag
                    
            else:
                feedFlag = 1
                print("no feed")
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    cap.release()


def lineCross(csvFile):
    global feedFlag, liveStream_frame, liveStream_name, event_occured, eventStream_path, eventStream_frame, totalCount,totalAttempts, livestream_loc

    check_location(os.path.join(os.getcwd(), "LineCrossing", str(datetime.now()).split(' ')[0]))
    camDetails = pd.read_csv(csvFile, delimiter=',')
    camList = []                                                # to store camera details from the database

    loc = 0  #variable to position the livestream windows on the screen
    
    ##linux version to get screen width and height##
    #screen = Window().get_screen()
    #width,height = screen.width(),screen.height()

    ##windows version to get screen width and height##
    #width, height = GetSystemMetrics(0), GetSystemMetrics(1) 
    
    threads = camDetails.shape[0]  # Number of threads to create
    
    
    for i in range(threads):
        reg = ([])
        line1 = camDetails['Line1'][i]
        line1 = [(line1.strip('()').split('), (')[0]), (line1.strip('()').split('), (')[1])]
        line2 = camDetails['Line2'].values[i]
        line2 = [((line2.strip('()').split('), (')[0])), ((line2.strip('()').split('), (')[1]))]
        roi = camDetails['Region'].values[i]
        
        roi = roi.replace('[','')
        roi = roi.replace(']','')
        roi = roi.split(',')

        for r in range(0,len(roi)-1,2):
            
            reg.append([int(roi[r]),int(roi[r+1])])
        data = camDetails.iloc[i].tolist()
        data[2] = line1
        data[3] = line2
        data[4] = reg
        camList.append(data)

    jobs = []       # to append all threads
    
    # to pass Name, Rtsp, Line1, Line2, ROI, case, slope, Livestream, Threshold to each thread from the csv
    for i in range(0, threads):
        
        if camList[i][7].lower() == "yes":
            loc +=1 
        
        thread = LineCrossing(camList[i][0], camList[i][1], camList[i][2], camList[i][3], camList[i][4], camList[i][5],
                              camList[i][6], camList[i][7], camList[i][8], (50+380*(loc-1),25))
        thread.setName(camList[i][0])
        jobs.append(thread)
    
    # To start threads appended in the list 'jobs'
    '''for lineThread in jobs:
        lineThread.start()
        time.sleep(2)'''
    jobs[1].start()
    
        
    print('Started all Threads')
    
    ###########livestream and event pop-up section################ 
    while True:
        
        try:
            if totalCount == threads*totalAttempts:         # if the no of restart attempts have exhusted for all threads, end program
                break
            
            if liveStream_name != None and  len(liveStream_frame.shape)>0:      #check status of global variables for livestreaming
                try:
                    cv2.namedWindow('Livestream for : {}'.format(liveStream_name))
                    cv2.moveWindow('Livestream for : {}'.format(liveStream_name),livestream_loc[0],livestream_loc[1])
                    cv2.imshow('Livestream for : {}'.format(liveStream_name), cv2.resize(liveStream_frame,(700,700)))
                    cv2.waitKey(1)
                    #if k == ord('q'):
                    #    cv2.destroyAllWindows()
                except Exception as e:
                    #cv2.destroyWindow('Livestream for : {}'.format(liveStream_name))            # if error in live stream show window, destroy that window
                    cv2.destroyAllWindows()
                    
            if eventStream_path != None and event_occured == True:
                eventStream_name = eventStream_path.split('\\')[-2]     # extract camera name from image path
    
                # create event text pop up
                t = time.localtime()
                currentTime = time.strftime("%H:%M:%S", t)
                camera_name = eventStream_name
                text = "Trespassing occurred on:"
    
                img = np.ones((150, 640,3))*(0,0,255)           # create a blank red window of width same as frame width
                cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,0), 1)
                cv2.putText(img, camera_name, (10,90), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1)
                cv2.putText(img, str(currentTime), (10,120), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1)
                
                img = img.astype(np.uint8, copy=False)
                eventImage = np.vstack((img,eventStream_frame))
    
                try:
                    for i in range(10):             # display event for this duration
                       if i == 0:
                           beep_thread = Alarm_sound(eventStream_name)
                           beep_thread.start()
                       else:
                           cv2.imshow("Warning at Camera: %s"%eventStream_name, cv2.resize(eventImage,(400,400)))
                           cv2.imshow('Livestream for : {}'.format(liveStream_name), cv2.resize(liveStream_frame,(400,400)))         # show livestream along with event
                           cv2.waitKey(1)                           
                    event_occured = False           # reset event flag
                except:                    
                    cv2.destroyWindow('Livestream for : {}'.format(liveStream_name))            # if error in live stream show window, destroy that window
                    feedFlag = 1
                    
        except KeyboardInterrupt:           # to exit script with ctrl+c command

            cv2.destroyAllWindows()
            quit()
            sys.exit()
    cv2.destroyAllWindows()

lineCross(os.getcwd()+"/LineCrossing/camera_details.csv")
#lineCross(os.getcwd()+"/LineCrossing2/final1.csv")
