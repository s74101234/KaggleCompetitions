from skimage import io,transform
import glob
import numpy as np
import re
import pandas as pd

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras import backend as K

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def f1_m(y_true, y_pred):
    THRESHOLD = 0.5 # 0.05
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

#createFilePD
def createFilePD(Path):
    Data1_columnNames = ["Id", 'Predicted']
    Data1 = pd.read_csv(Path, names = Data1_columnNames, skiprows = 1)
    FileName_Data = Data1['Id']
    FileName_Data = pd.DataFrame(FileName_Data) 
    FileName_Data = FileName_Data + '.png'
    FileName_Data.columns = ['Filenames']
    return FileName_Data, Data1

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 224, 224, 3 #224, 224 , 3
    batch_size = 12
    readDataPath = "./Data/test/sample_submission.csv"
    DataDirPath = "./Data/test/test2"
    loadModelPath = "./Model_SEResnet50/Keras_19_0.0943_0.2700.h5"
    writePath = "./Model_SEResnet50/result_19_0.0943_0.2700_0.2.csv"

    #載入資料
    data, FileName = createFilePD(readDataPath)
    print(data.head(5))

    test_datagen = ImageDataGenerator(rescale=1./255.)

    test_generator=test_datagen.flow_from_dataframe(
        dataframe=data,
        directory=DataDirPath,
        x_col="Filenames",
        batch_size=batch_size,
        shuffle=False,
        class_mode=None,
        target_size=(img_height,img_width))

    #載入模型
    model = load_model(loadModelPath, custom_objects={'f1_m': f1_m})
    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    fw = open(writePath, "w")
    fw.write('Id,Predicted\n')
    for idx in range(0, pred.shape[0], 1):
        result = ""
        Temp = np.where(pred[idx]>0.2)[0]#0.2
        for idx2 in range(0, len(Temp), 1):
            result += str(Temp[idx2]) + " "
        result = result[:-1]
        print('%s,%s'%(FileName['Id'][idx], str(result)))
        fw.write('%s,%s\n'%(FileName['Id'][idx], str(result)))
    fw.close()