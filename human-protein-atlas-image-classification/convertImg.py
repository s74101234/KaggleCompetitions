import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image

if __name__ == "__main__":
    #參數設定
    # readDataPath = "./Data/train/train_Temp.csv"
    # DataDirPath = "./Data/train/train_Temp/"
    # DataDirPath2 = "./Data/train/train2_Temp/"

    readDataPath = "./Data/train/train.csv"
    DataDirPath = "./Data/train/train/"
    DataDirPath2 = "./Data/train/train3/"

    # readDataPath = "./Data/test/sample_submission.csv"
    # DataDirPath = "./Data/test/test/"
    # DataDirPath2 = "./Data/test/test2/"

    Data1_columnNames = ["Id","Target"]
    colors = ['red', 'green', 'blue']
    # colors = ['green', 'blue', 'red' ,'yellow']
    Data1 = pd.read_csv(readDataPath, names = Data1_columnNames, skiprows = 1)
    for FileName in Data1["Id"]:
        print('read File：', FileName)
        # flags = cv2.IMREAD_GRAYSCALE
        # img = []
        # for color in colors:
        #     img.append(cv2.imread(os.path.join(DataDirPath, FileName+'_'+color+'.png'), flags))
        # img = cv2.merge(img,len(img))

        # Function 1
        img = []
        for color in colors:
            Temp = Image.open(os.path.join(DataDirPath, FileName+'_'+color+'.png')) 
            Temp = np.asarray(Temp)
            img.append(Temp)
        img = np.stack(img, axis=-1)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(DataDirPath2 + FileName + '.png', img) 
        
        # # Function 2
        # Red = Image.open(os.path.join(DataDirPath, FileName+'_'+colors[0]+'.png')) 
        # Green = Image.open(os.path.join(DataDirPath, FileName+'_'+colors[1]+'.png')) 
        # Blue = Image.open(os.path.join(DataDirPath, FileName+'_'+colors[2]+'.png')) 
        # img = np.stack((np.array(Red), np.array(Green), np.array(Blue)), -1)
        # img = cv2.resize(img, (224, 224))
        # cv2.imwrite(DataDirPath2 + FileName + '.png', img) 
    
