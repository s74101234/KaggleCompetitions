import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, ZeroPadding2D, add
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import f1_score

# SENet 
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras import backend as K
import tensorflow as tf

# Micro F1 Score
# One Label
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     # recall = true_positives / possible_positives
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     # precision = true_positives / predicted_positives
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
#     # return 2 * (precision * recall)/(precision + recall)

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

#參考 https://blog.csdn.net/wmy199216/article/details/71171401
#參考 https://github.com/titu1994/keras-squeeze-excite-network
def squeeze_excite_block(input, ratio = 16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias = False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias = False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def Conv2d_BN(x, nb_filter, kernel_size, strides = (1,1), padding = 'same'):
    x = Conv2D(nb_filter,kernel_size, padding = padding, strides = strides, activation = 'relu')(x)
    x = BatchNormalization(axis = 3)(x)
    return x
 
def Conv_Block_2(inputs, nb_filter, kernel_size, strides = (1, 1), with_conv_shortcut = False):
    x = Conv2d_BN(inputs, nb_filter = nb_filter, kernel_size = kernel_size, strides = strides, padding = 'same')
    x = Conv2d_BN(x, nb_filter = nb_filter, kernel_size = kernel_size, padding = 'same')
    x = squeeze_excite_block(x)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inputs, nb_filter = nb_filter, strides = strides, kernel_size = kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inputs])
        return x
 
def Conv_Block_3(inputs, nb_filter, kernel_size, strides = (1, 1), with_conv_shortcut = False):
    x = Conv2d_BN(inputs, nb_filter = nb_filter[0], kernel_size = (1, 1), strides = strides, padding='same')
    x = Conv2d_BN(x, nb_filter = nb_filter[1], kernel_size = (3,3), padding = 'same')
    x = Conv2d_BN(x, nb_filter = nb_filter[2], kernel_size = (1,1), padding = 'same')
    x = squeeze_excite_block(x)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inputs, nb_filter = nb_filter[2], strides = strides, kernel_size = kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inputs])
        return x

def buildSE_ResNet34Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3,3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildSE_ResNet50Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    # sigmoid_cross_entropy_with_logits, binary_crossentropy
    model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildSE_ResNet101Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model

def buildSE_ResNet152Model(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',#0.001
              metrics = ['categorical_accuracy', 'binary_accuracy', f1_m])
    return model
