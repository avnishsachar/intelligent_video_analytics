## LineCross event detection with blinking pop up ##
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
import tkinter as tk
from tkinter import *
import tkinter.messagebox 


## SEND EMAIL ALERTS
import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class LineCrossing(threading.Thread):

    def __init__(self, name, rtsp, line1, line2, region, case, slope, livestream, threshold):

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
        #self.frame_queue = frame_queue

    def run(self):
        detectPeople(self.NAME, self.RTSP, self.LINE1, self.LINE2, self.REGION, self.CASE, self.SLOPE, self.LIVESTREAM,
                     self.THRESHOLD)
        count = 0
        #restart process if feedloss occurs
        while count<10:
            
            if feedFlag == 1:
                print("restarted",self.NAME,count)
                count+=1
                
                detectPeople(self.NAME, self.RTSP, self.LINE1, self.LINE2, self.REGION, self.CASE, self.SLOPE,
                             self.LIVESTREAM, self.THRESHOLD)

    def get_id(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                print(thread)
                return id


def start_eventStream(name, rtsp):
    cap = cv2.VideoCapture(rtsp)
    print("event occured",name,rtsp)
    j = 0
    while True:
        ret,frame = cap.read()
        while j < 300 and ret == True:
            cv2.imshow(name,frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            j+=1
    cv2.destroyAllWindows()
    cap.release()
## start_eventStream yet to be made
class EventStream(threading.Thread):

    def __init__(self, rtsplink):
        threading.Thread.__init__(self)
        self.NAME = rtsplink[1]
        self.RTSP = rtsplink[0]

    def run(self):
        start_eventStream(self.NAME, self.RTSP)


def email_alert(name, rtsp, reason):
    print('sending email alert')
    sender = "sujayrpi@gmail.com"
    receiver = "rishabh@omnipresenttech.com"

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['to'] = receiver
    msg['Subject'] = "[	ATTENTION USER	]"
    body = " Your camera '{}' of RTSP {} has encountered {}".format(name, rtsp, reason)
    msg.attach(MIMEText(body, 'plain'))

    try:
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security
        s.starttls()
        # Authentication
        s.login(sender, "sujay2908")
        print("Logging into server account")

        # Converts the Multipart msg into a string
        text = msg.as_string()
        # for i in receiver:
        #	print(i)
        s.sendmail(sender, receiver, text)
        # terminating the session
        s.quit()
    except:
        print("Email not sent, No Internet Connection")
        pass


# Function to create the folders
def check_location(location):
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except:
            print('Unable to create directory')


# to check if event occurred or not
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

# implementation of line crossing
def detectPeople(name, rtsplink, line1, line2, region, case, slope, livestream, threshold):
    global direc, feedFlag, liveStream_frame, liveStream_name, event_occured, eventStream_path, eventStream_frame
    frame_queue = queue.Queue(maxsize=0)
    # flags initialisation
    eventFlag = False                       #If event occurs
    liveStream_frame = None                 #camera frame if livestream is on
    liveStream_name = None                  #camera name if livestream is on
    event_occured = False
    feedFlag = 0                            #Flag to check feedloss
    tresFlag = 0                            #Flag to check trespassing
    countFrame = 0                          #Count to skip frames
    eventStream_frame = None
    eventStream_path = None

    check_location(os.path.join(os.getcwd(), "LineCrossing", str(datetime.now()).split(' ')[0], name))
    # Input_vid = 'filesrc location=3.ts ! tsdemux ! queue ! h264parse ! nvv4l2decoder!  nvvidconv ! video/x-raw, format=RGBA, height=480, width=640! videoconvert ! appsink'
    # Input_vid = "rtspsrc location="+rtsplink+"! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
    #Input_vid = "rtspsrc location=" + rtsplink + " ! rtph264depay ! h264parse ! nvv4l2decoder !  nvvidconv ! video/x-raw, format=RGBA, height=480, width=640! videoconvert ! appsink"
    Input_vid = rtsplink
    cap = cv2.VideoCapture(Input_vid)

    line1 = (int(line1[0].split(',')[0]), int(line1[0].split(',')[1]), int(line1[1].split(',')[0]), int(line1[1].split(',')[1]))
    line2 = (int(line2[0].split(',')[0]), int(line2[0].split(',')[1]), int(line2[1].split(',')[0]), int(line2[1].split(',')[1]))

    if slope != 'null':
        const1 = line1[1] - slope * line1[0]  # to find eq of lines (y = mx + const)
        const2 = line2[1] - slope * line2[0]
    R_x1, R_y1 = int(region[0]), int(region[1])  # ROI points
    R_x2, R_y2 = int(region[0] + region[2]), int(region[1] + region[3])

    backSub = cv2.createBackgroundSubtractorMOG2()
    backSub.setVarThreshold(12)

    while True:

        ret,frame_org = cap.read()
        countFrame += 1
        check_location(os.path.join(os.getcwd(),"LineCrossing",str(datetime.now()).split(' ')[0],name))
        if countFrame%3==0:  # to skip frames
            if ret == True:
                frame_org = cv2.resize(frame_org,(640,480))
                frame = frame_org[R_y1:R_y2,R_x1:R_x2]

                cv2.rectangle(frame_org, (R_x1, R_y1), (R_x2, R_y2), (255, 0, 0), 2)
                cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 255), 2)
                cv2.line(frame, (line2[0], line2[1]), (line2[2], line2[3]), (0, 0, 255), 2)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                thresh = backSub.apply(frame,learningRate=0.7)

                # find contours of detected objects
                (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in cnts:
                    if cv2.contourArea(c) < threshold:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)

                    lwrCtr = (int(x + w / 2), (y + h))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, lwrCtr, 1, (255, 255, 255), 5)

                    if slope != 'null':
                        dist = ((slope) * (lwrCtr[0]) - (lwrCtr[1]))
                        d1 = (dist + (const1))
                        d2 = (dist + (const2))
                    else:
                        d1 = lwrCtr[0] - line1[0][0]
                        d2 = lwrCtr[0] - line2[0][0]

                    tresFlag = checkCase(d1, d2, case)

                if tresFlag == 1:
                    cv2.putText(frame, "WARNING", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    tresFlag = 0

                elif tresFlag == 2:
                    cv2.putText(frame, "CRITICAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Save frame with time for critical event
                    frame_name = str(datetime.now().strftime("%H-%M-%S"))
                    today = str(date.today())
                    if not os.path.exists(os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name)):
                        print('Saving Frame', name)
                        cv2.imwrite(os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name), frame_org)
                        image_path = os.path.join(os.getcwd(), "LineCrossing", today, name,'Tresspass_alert_%s.jpg' % frame_name)
                        
                        frame_queue.put(image_path)
                        eventFlag = True
                        tresFlag = 0

                else:
                    cv2.putText(frame, "OK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if livestream == 'Yes':
                    liveStream_name = name
                    liveStream_frame = frame_org

                if eventFlag == True and not frame_queue.empty():
                    
                    eventStream_path = frame_queue.get()
                    eventStream_frame = frame_org
                    event_occured = True
                
            else:
                #feedFlag = 1
                break

    cv2.destroyAllWindows()
    cap.release()
    
def openFile(eventpath):
    os.system('eog '+ eventpath)

def lineCross(csvFile):
    global feedFlag, liveStream_frame, liveStream_name, event_occured, eventStream_path, eventStream_frame

    check_location(os.path.join(os.getcwd(), "LineCrossing", str(datetime.now()).split(' ')[0]))
    camDetails = pd.read_csv(csvFile, delimiter=',')
    camList = []                                                # to store camera details from the database
    
    threads = camDetails.shape[0]  # Number of threads to create
    for i in range(threads):
        line1 = camDetails['Line1'][i]
        line1 = [((line1.strip('()').split('), (')[0])), ((line1.strip('()').split('), (')[1]))]
        line2 = camDetails['Line2'].values[i]
        line2 = [((line2.strip('()').split('), (')[0])), ((line2.strip('()').split('), (')[1]))]
        roi = camDetails['Region'].values[i]
        roi = roi.strip('()').split(', ')
        roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
        data = camDetails.iloc[i].tolist()
        data[2] = line1
        data[3] = line2
        data[4] = roi
        camList.append(data)

    jobs = []       # to append all threads
    event_jobs = []
    
    # to pass Name, Rtsp, Line1, Line2, ROI, case, slope, Livestream, Threshold to each thread
    for i in range(0, threads):
        thread = LineCrossing(camList[i][0], camList[i][1], camList[i][2], camList[i][3], camList[i][4], camList[i][5],
                              camList[i][6], camList[i][7], camList[i][8])
        thread.setName(camList[i][0])
        jobs.append(thread)
    
    # To start threads appended in the list 'jobs'
    for lineThread in jobs:
        lineThread.start()
        print(lineThread.getName())
        time.sleep(2)

    print('Started all Threads')

    # To display livestream in case of event or if it is user enabled
    while True:

        if liveStream_name != None:
            cv2.imshow('Livestream for : {}'.format(liveStream_name), liveStream_frame)
            cv2.waitKey(1)
            #if k == ord('q'):
            #   break

        if eventStream_path != None and event_occured == True:
            print(eventStream_path)
            eventStream_name = eventStream_path.split('\\')[-2]
            
            #create event text pop up
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            camera_name = eventStream_name + " at "+str(current_time)
            text = "Trespassing occurred on:"
            
            width1,height1 = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, thickness=1)[0]
            width2,height2 = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=1)[0]
            
            max_wid = max(width1,width2)
            img = np.ones((100, max_wid+20,3))*(0,0,255)    #create a blank red window of adjustable width
            
            cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,0,0), 1)
            cv2.putText(img, camera_name, (10,90), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1)
            cv2.imshow('Livestream for : {}'.format(liveStream_name), liveStream_frame)         #show livestream along with event
            cv2.waitKey(1)
                
            for i in range(10):             #show the blinking text window 
                cv2.imshow("Warning at %s!!"%eventStream_name, img)
                cv2.waitKey(1000)
                cv2.destroyWindow("Warning at %s!!"%eventStream_name)
            
            #show event frame for 10s 
            j = 0
            cv2.namedWindow('Event Occured on : {}'.format(eventStream_name))
            cv2.moveWindow('Event Occured on : {}'.format(eventStream_name), 300,300)
            cv2.resizeWindow('Event Occured on : {}'.format(eventStream_name),480,320)
            while j<10:
                print('Count no : ',j)
                cv2.imshow('Event Occured on : {}'.format(eventStream_name), eventStream_frame)
                cv2.imshow('Livestream for : {}'.format(liveStream_name), liveStream_frame)         #show livestream along with event
                cv2.waitKey(1)
                k = cv2.waitKey(1)
                time.sleep(1)
                if k == ord('q'):
                    break
                j+=1
            event_occured = False
            cv2.destroyWindow('Event Occured on : {}'.format(eventStream_name))
            
    cv2.destroyAllWindows()

lineCross(os.getcwd()+"/LineCrossing/camera_details_ts.csv")


