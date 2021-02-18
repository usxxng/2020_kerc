# -*- coding: utf-8 -*-

import os
import pandas as pd

col_name = ["class", "file_name", "cl1", "cl2"]
file1 = pd.read_csv("Data/data_information.csv", names= col_name)
file2 = pd.read_csv("csv/train_mean.csv")
file3 = pd.read_csv("csv/val_mean.csv")

data =[]

for i in range (len(file1)):
    class_name = file1["class"].values[i]
    file_name = file1["file_name"].values[i]
    cl1 = file1["cl1"].values[i]
    cl2 = file1["cl2"].values[i]
    elif(class_name=="Train"):i
        v = file2["Valence"].values[i]
        a = file2["Arousal"].values[i]
        s = file2["Stress"].values[i]
    if(class_name=="Val"):
        tt = i-len(file2)
        # import pdb; pdb.set_trace()
        v = file3["Valence"].values[tt]
        a = file3["Arousal"].values[tt]
        s = file3["Stress"].values[tt]
    
    row = []
    row.append(class_name)
    row.append(file_name)
    
    row.append(v)
    row.append(a)
    row.append(s)
    row.append(cl1)
    row.append(cl2)
    data.append(row)
            
df = pd.DataFrame(data)
df.to_csv("Data/data_file.csv", index= None, header = None)
