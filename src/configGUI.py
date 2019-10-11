## GUI to add a camera to the database ##

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:41 2019

@author: Asus
"""

try:
   from Tkinter import *
   import Tkinter as tk
   import tkMessageBox
except ImportError:
   from tkinter import *
   import tkinter as tk
   import tkinter.messagebox 

import numpy as np
import os
import cv2
import csv
import time

from markRegion import *


def configure():
    global camVal,UIVal,psdVal,IPVal,ChVal,LSVal,file_path,Tb,cam_info


    if camVal.get() == '' or UIVal.get() == '' or psdVal.get() == '' or IPVal.get() == '': 
        tkinter.messagebox.showinfo("Error","Enter All Fields!")
	#tkMessageBox.showinfo("Error","Enter All Fields!")
        pass
        
    else:
        cam = (camVal.get())
        UI = (UIVal.get())
        psd = (psdVal.get())
        IP = (IPVal.get())
        if ChVal.get() == '':
            Ch = "101"
        else:
            Ch = (ChVal.get())
        if LSVal.get() == '':
            LS = 'no'
        else:
            LS = (LSVal.get())

        vid_url = "rtsp://" + UI + ':' + psd + '@' + IP + "/Streaming/Channels/"+ Ch
        info = markRoi(vid_url)
        
        if info =="null_rtsp":
            #tkMessageBox.showinfo("Error","Could not connect to camera.\n            Try again!")
            tkinter.messagebox.showinfo("Error!","Could not connect to camera.\n             Try again!")
            
        elif info == "null_pts":
            tkinter.messagebox.showinfo("Error!","Wrong Input, Try Again.")
            #info = markRoi(vid_url)
            configure()
        else:
            cam_info = [cam,vid_url]+info+[LS]
            tkinter.messagebox.showinfo("Message","Add Threshold for the camera.")
            Tb['state'] = 'normal'
            


def quitWindow():
    global win2, threshVal,cam_info,Tb
    
    th = (threshVal.get())
    cam_info_new = cam_info+[th]
    with open(file_path, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(cam_info_new) 
    win2.destroy()
    tkinter.messagebox.showinfo("Info","Camera Added!")
    Tb['state'] = 'disabled'
        
def thresh():
    global win2,threshVal,croot
    
    win2 = tk.Toplevel(croot)
    win2.title("Threshold")
    win2.geometry("350x150+150+200")
    text = tk.Label(win2, text="Threshold for the:\n camera")
    text.place(relx = 0.2,rely = 0.3,anchor = CENTER)
            
    threshVal = tk.Entry(win2, width = 20)
    threshVal.place(relx=0.7,rely=0.3,anchor=CENTER)
    Ob = tk.Button(win2, text="Ok", bg = "gray",fg = "orange", bd=2,font = 'Calibri 14 bold',relief = 'groove',width=5,command = quitWindow)
    Ob.place(relx=0.5,rely=0.7,anchor=CENTER)

def config():
#if __name__ == "__main__":    
    global camVal,UIVal,psdVal,IPVal,ChVal,LSVal,file_path,Tb,croot
    
    #croot.iconbitmap(default=r"0.ico")
    croot = tk.Tk()
    croot.title("ADD CAMERAS")
    croot.geometry("600x400+50+100")
    frame=tk.Frame(croot)
    frame.pack()

    try:
        os.makedirs(folderName,exist_ok=True)
    except :#FileExistsError:
        
        pass
    
    file_path = os.path.join(os.getcwd(),"LineCrossing","camera_details.csv")
    if not os.path.exists(file_path):
        field = ['Name','URL','Line1','Line2','Region','Case','Slope','LiveStream','Threshold']
        with open(file_path, "w+") as csv_camInfo:
            writer =  csv.writer(csv_camInfo)
            writer.writerow(field)
    else:
        pass
    
    ReqLabel= tk.Label(croot, text="* Required fields", fg = "red")#,bg = 'light steel blue4', bd=2,font = 'Calibri 11 bold')
    ReqLabel.place(relx=0.1,rely=0.1,anchor=W)
    
    camLabel= tk.Label(croot, text="Camera Name:", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    camLabel.place(relx=0.1,rely=0.2,anchor=W)
    camLabel= tk.Label(croot, text="*", fg = "red")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    camLabel.place(relx=0.3,rely=0.2,anchor=W)

    UILabel = tk.Label(croot, text="User ID:", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold',anchor= 'w')
    UILabel.place(relx=0.1,rely=0.3,anchor=W)
    UILabel = tk.Label(croot, text="*", fg = "red")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold',anchor= 'w')
    UILabel.place(relx=0.21,rely=0.3,anchor=W)

    psdLabel = tk.Label(croot, text="Password:", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    psdLabel.place(relx=0.1,rely=0.4,anchor=W)
    psdLabel = tk.Label(croot, text="*", fg = "red")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    psdLabel.place(relx=0.24,rely=0.4,anchor=W)
    
    IPLabel= tk.Label(croot, text="IP Address:", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    IPLabel.place(relx=0.1,rely=0.5,anchor=W)
    IPLabel= tk.Label(croot, text="*", fg = "red")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    IPLabel.place(relx=0.25,rely=0.5,anchor=W)

    ChLabel = tk.Label(croot, text="Channel:", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    ChLabel.place(relx=0.1,rely=0.6,anchor=W)
    
    LSLabel = tk.Label(croot, text="Live Stream (yes/no):", fg = "black")#,bg = 'light steel blue4', bd=2,font = 'Calibri 14 bold')
    LSLabel.place(relx=0.1,rely=0.7,anchor=W)

    camVal = tk.Entry(croot,width=40)
    camVal.place(relx=0.7,rely=0.2,anchor=CENTER)
    
    UIVal = tk.Entry(croot,width=40)
    UIVal.place(relx=0.7,rely=0.3,anchor=CENTER)

    psdVal = tk.Entry(croot,width=40)
    psdVal.place(relx=0.7,rely=0.4,anchor=CENTER)
    
    IPVal = tk.Entry(croot,width=40)
    IPVal.place(relx=0.7,rely=0.5,anchor=CENTER)
    
    c = tk.StringVar(croot, value='101')
    ChVal = tk.Entry(croot,width=40,fg = "gray",textvariable = c)
    ChVal.place(relx=0.7,rely=0.6,anchor=CENTER)
    
    v = tk.StringVar(croot, value='No')
    LSVal = tk.Entry(croot, width = 40,textvariable=v,fg = "gray")
    LSVal.place(relx=0.7,rely=0.7,anchor=CENTER)

    Cb = tk.Button(croot,text ="CONFIGURE CAMERA",bg = "gray",fg = "orange", bd=2,font = 'Calibri 14 bold', command= configure)#,relief = 'groove')
    Cb.place(relx=0.3, rely=0.85, anchor=CENTER)

    Tb = tk.Button(croot,text ="Add Threshold",bg = "gray",fg = "orange", bd=2,font = 'Calibri 14 bold', command= thresh,state=DISABLED)#,relief = 'groove')
    Tb.place(relx=0.7, rely=0.85, anchor=CENTER)
        
    croot.mainloop()

#config()