## starts the MAIN GUI application ##

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 23:22:28 2019

@author: Asus
"""

from tkinter import *
from tkinter import ttk
import tkinter as tk
import tkinter.messagebox 
import os
import pandas as pd

from configGUI import *
#from lineCrossThread import *
from deleteGUI import *
from displayGUI import *

root = tk.Tk()

def Add():
    global root,folderName
    root.destroy()
    config()
    root=Tk() 
    MainGui()         

def Remove():
    global root,file_path
    root.destroy()
    deleteCam(file_path)
    root=Tk() 
    MainGui()
   

def Process():
    global root,file_path
    root.destroy()
    if not os.path.exists(file_path):
        tkinter.messagebox.showinfo("Error!","Please Add Cameras")
    else:
        lineCross(file_path)

def ShowCam():
    global root,file_path
    root.destroy()
    Display(file_path)
    root=Tk() 
    MainGui()
    
def MainGui():
    global root, file_path

    root.title("Mange Cameras")
    root.geometry("300x250+50+100")
    frame=tk.Frame(root)
    #root.iconbitmap(default=r"0.ico")

    folderName = "LineCrossing"
    file_path = folderName + "/camera_details.csv"
    frame.pack()
        
    add = tk.Button(root, text='Add Cameras', width=25,height =1,fg = "orange",bg = "gray", bd=2,font = 'Calibri 14 bold', command = Add,relief="raised")
    add.pack(fill = BOTH,padx = 20,pady = 12)
    
    remove = tk.Button(root, text='Remove Cameras', width=25,height =1,fg = "orange",bg = "gray", bd=2, font = 'Calibri 14 bold',command = Remove,relief="raised")
    remove.pack(fill = BOTH,padx = 20,pady = 12)
    
    process = tk.Button(root, text='Start Line Crossing', width=25,height =1,fg = "orange",bg = "gray", bd=2,font = 'Calibri 14 bold', command = Process,relief="raised")
    process.pack(fill = BOTH,padx = 20,pady = 12)
    
    show = tk.Button(root, text='Show Cameras', width=25,height =1,fg = "orange",bg = "gray", bd=2,font = 'Calibri 14 bold', command = ShowCam,relief="raised")
    show.pack(fill = BOTH,padx = 20,pady = 12)
    
    root.mainloop()

MainGui()