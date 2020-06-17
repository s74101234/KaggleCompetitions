import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet201
import tensorflow as tf

# Macro F1 Score
# 參考 https://www.kaggle.com/wordroid/inceptionresnetv2-resize256-f1loss-lb0-419
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

def f1_loss(y_true, y_pred):
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def buildMobileNetV2Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    # --------------------------------------------
    StopNum = 0
    AppModel = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')
    for idx, layer in enumerate(AppModel.layers):
        layer.trainable = False
        if(idx == StopNum):
            break
    x = AppModel.layers[-1].output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation = 'sigmoid', name = 'sigmoid')(x)
    model = Model(inputs = AppModel.input, outputs = output_layer)
    # --------------------------------------------
    model.summary()    
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(lr=1e-4),#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildResNet50Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    # --------------------------------------------
    StopNum = 0
    AppModel = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    for idx, layer in enumerate(AppModel.layers):
        layer.trainable = False
        if(idx == StopNum):
            break
    x = AppModel.layers[-1].output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation = 'sigmoid', name = 'sigmoid')(x)
    model = Model(inputs = AppModel.input, outputs = output_layer)
    # --------------------------------------------
    model.summary()    
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(lr=1e-4),#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildDenseNet121Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    # --------------------------------------------
    StopNum = 0
    AppModel = DenseNet121(include_top=False, pooling='avg', weights='imagenet')
    for idx, layer in enumerate(AppModel.layers):
        layer.trainable = False
        if(idx == StopNum):
            break
    x = AppModel.layers[-1].output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation = 'sigmoid', name = 'sigmoid')(x)
    model = Model(inputs = AppModel.input, outputs = output_layer)
    # --------------------------------------------
    model.summary()    
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(lr=1e-4),#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildDenseNet201Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    # --------------------------------------------
    StopNum = 0
    AppModel = DenseNet201(include_top=False, pooling='avg', weights='imagenet')
    for idx, layer in enumerate(AppModel.layers):
        layer.trainable = False
        if(idx == StopNum):
            break
    x = AppModel.layers[-1].output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation = 'sigmoid', name = 'sigmoid')(x)
    model = Model(inputs = AppModel.input, outputs = output_layer)
    # --------------------------------------------
    model.summary()    
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(lr=1e-4),#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model
