from skimage import io,transform
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import keras
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#createFilePD
def createFilePD(Path, classes, img_height, img_width, img_channl):
    Data1_columnNames = ["Id","Target"]
    Data1 = pd.read_csv(Path, names = Data1_columnNames, skiprows = 1)

    FileName_Data = Data1['Id']
    FileName_Data = pd.DataFrame(FileName_Data) 
    FileName_Data = FileName_Data + '.png'

    Labels = []
    for idx in range(0, len(Data1['Target']), 1):
        Label = Data1['Target'][idx]
        Labels.append(Label.split(' '))

    # Convert OneHot Encode
    # Labels = Data1['diagnosis'].to_numpy().reshape(-1, 1)
    # OneHot_Labels = np_utils.to_categorical(Labels, len(classes))
    # OneHot_df = pd.DataFrame(OneHot_Labels).astype('int32')

    # Convert Multi Label
    MB = MultiLabelBinarizer(classes=classes)
    ML_Labels = MB.fit_transform(Labels)
    ML_Labels = pd.DataFrame(ML_Labels)

    Data_concat = pd.concat([FileName_Data, ML_Labels], axis=1)
    Data_concat.columns = ['Filenames'] + classes
    # print(Data_concat)
    return Data_concat

def saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size,
                    train, val, DataDirPath, img_height, img_width, columns):
	# DataGen
    train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=DataDirPath,
        x_col="Filenames",
        y_col=columns,
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='other')
        
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=DataDirPath,
        x_col="Filenames",
        y_col=columns,
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='other')

    #設置TensorBoard
    tbCallBack = TensorBoard(log_dir = saveTensorBoardPath, batch_size = batch_size,
                            write_graph = True, write_grads = True, write_images = True,
                            embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
                            monitor = 'val_loss', verbose = 1, 
                            save_best_only = True, mode = 'min',
                            filepath = ('%s_{epoch:02d}_{val_loss:.4f}_{val_f1_m:.4f}.h5' %(saveModelPath)))

    #設置ReduceLROnPlateau
    Reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 3, cooldown = 1, verbose = 1, mode = 'min')

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_loss', patience = 9, verbose = 1, mode = 'min')

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit_generator(train_generator,
                steps_per_epoch = len(train)//batch_size,
                epochs = epochs,
                shuffle = True,
                validation_data = val_generator,
                validation_steps = len(val)//batch_size, 
                callbacks = callbacks_list)
    