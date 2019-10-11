## Remove selected cameras from the csv database ##

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:20:46 2019

@author: Asus
"""

import pandas as pd
import csv
import numpy as np

def configDelete(to_delete,csv_file):
    
    df = pd.read_csv(csv_file)
    for cam in to_delete:
        df.drop(df.loc[df['Name']==cam].index, inplace=True)
        
    final_data = []
    length = len(df.values)
    for i in range(length):
        data =  np.array([])
        x_loc  =  df.iloc[i, :].values
        data  =  np.append(data, x_loc)
        final_data.append(data)
    
    header = list(df.columns.values) 
    with open(csv_file, "w") as csv_file:
        writer =  csv.writer(csv_file, lineterminator = '\n')
        writer.writerow(header)
        writer.writerows(final_data)
    