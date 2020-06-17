from skimage import io, transform
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import keras
from keras.utils import np_utils
from core.main import createFilePD, saveTrainModels_gen
from core.Model.LeNet import buildLeNetModel
# from core.Model.ResNet import buildResNet34Model, buildResNet50Model, buildResNet101Model, buildResNet152Model
from core.Model.SE_ResNet import buildSE_ResNet34Model, buildSE_ResNet50Model, buildSE_ResNet101Model, buildSE_ResNet152Model
from core.Model.KerasApplication import buildDenseNet121Model, buildDenseNet201Model, buildMobileNetV2Model
from core.Model.KerasApplication import buildResNet50Model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 224, 224, 3 #224, 224, 3
    classes = ["0","1","2","3","4","5","6","7","8","9",
            "10","11","12","13","14","15","16","17","18","19",
            "20","21","22","23","24","25","26","27"]
    num_classes = len(classes)
    batch_size = 24
    epochs = 10000
    dataSplitRatio = 0.8
    readDataPath = "./Data/train/train.csv"
    DataDirPath = "./Data/train/train2"
    saveModelPath = "./Model_AppResnet50/Keras"
    saveTensorBoardPath = "./Model_AppResnet50/Tensorboard/"
    num_GPU = 1

    #載入資料
    data = createFilePD(readDataPath, classes, img_height, img_width, img_channl)
    print(data.head(5))

    #切割資料
    num_example = data.shape[0]
    s = np.int(num_example * dataSplitRatio)
    train = data[:s]
    val = data[s:]

    print('train shape : ', train.shape)
    print('val shape : ', val.shape)
    
    # 建構模型
    # model = buildSE_ResNet34Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildSE_ResNet101Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildSE_ResNet152Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildLeNetModel(img_height, img_width, img_channl, num_classes, num_GPU)

    
    # model = buildMobileNetV2Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildSE_ResNet50Model(img_height, img_width, img_channl, num_classes, num_GPU)
    model = buildResNet50Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildDenseNet121Model(img_height, img_width, img_channl, num_classes, num_GPU)
    # model = buildDenseNet201Model(img_height, img_width, img_channl, num_classes, num_GPU)

    
    #訓練及保存模型
    saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, train, val, DataDirPath, img_height, img_width, classes)