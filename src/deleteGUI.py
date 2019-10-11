## GUI to remove cameras from database ##

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:57:54 2019

@author: Asus
"""

from tkinter import *
from tkinter import ttk
import tkinter as tk
import tkinter.messagebox 
import pandas as pd
from delete_row import*

def delete():
    global _string,droot,csv_file

    configDelete(_string,csv_file)
    tkinter.messagebox.showinfo("INFO","Camera(s) Removed!")
    droot.destroy()

def update_text():
    global cbs,_string
    
    _string = []
    for name, checkbutton in cbs.items():
        if checkbutton.var.get():
            _string.append(checkbutton['text'])

def deleteCam(filepath):
    global droot,csv_file,cbs
    droot = tk.Tk()
    
    droot.title("DELETE CAMERAS")

    vscrollbar = Scrollbar(droot)
    vscrollbar.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(droot,yscrollcommand=vscrollbar.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)

    vscrollbar.config(command=canvas.yview)
    frame = Frame(canvas)
    frame.rowconfigure(1, weight=1)
    frame.columnconfigure(1, weight=1)
    slctLabel= tk.Label(frame, text="Select cameras to be removed:", fg = "black",font = 'Calibri 11 bold')
    slctLabel.grid(column=1,pady = 5,padx=10)
    
    csv_file = filepath
    #csv_file = r"D:\FACE_DETECT_RECOG\video_details.csv"
    df = pd.read_csv(csv_file)
    
    names = (df['Name'].values)

    cbs = dict()
    for i, value in enumerate(names):
        cbs[value] = tk.Checkbutton(frame, text=value, onvalue=True,
                                offvalue=False, command=update_text)
        cbs[value].var = tk.BooleanVar(frame, value=False)
        cbs[value]['variable'] = cbs[value].var

        cbs[value].grid(column=1, row=i+2, sticky='W', padx=20)

    dltButton = Button(droot, text='Delete Cameras', width=25,height =1,fg = "orange",bg = "gray", bd=2,font = 'Calibri 14 bold', command = delete,relief="raised")
    dltButton.grid(column =0,padx = 20,pady=20)
    canvas.create_window(0, 0, anchor=NW, window=frame)
    frame.update_idletasks()

    canvas.config(scrollregion=canvas.bbox("all"))

    droot.mainloop()