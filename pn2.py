import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2 as cv
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
from keras.utils import plot_model
from keras.layers import Input, Add, Activation, BatchNormalization, Conv2D,\
     DepthwiseConv2D, UpSampling2D, MaxPooling2D, Dropout

def binary_focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    eps = backend.epsilon()
    pt_1 = backend.clip(pt_1, eps, 1. - eps)
    pt_0 = backend.clip(pt_0, eps, 1. - eps)
    return -(backend.mean(alpha * backend.pow(1. - pt_1, gamma) * backend.log(pt_1), axis=-1) \
        + backend.mean((1 - alpha) * backend.pow(pt_0, gamma) * backend.log(1. - pt_0), axis=-1))

def D_Block(input_layer, filts, isConcat=False, concat_layer=None):
    # Is the bottom block?
    if isConcat == True and concat_layer is not None:
        bl = Add()([input_layer, concat_layer])
    else:
        bl = input_layer
    # Left conv 1x1
    bl_left_route = Conv2D(filters=filts, kernel_size=1, strides=1, padding='same', data_format='channels_last', use_bias=False)(bl)
    bl_left_route = BatchNormalization()(bl_left_route)
    # Right conv dw 3x3
    bl_right_route = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False)(bl)
    bl_right_route = BatchNormalization()(bl_right_route)
    bl_right_route = Activation('relu')(bl_right_route)
    # Right conv 1x1
    bl_right_route = Conv2D(filters=filts, kernel_size=1, strides=1, padding='same', data_format='channels_last', use_bias=False)(bl_right_route)
    bl_right_route = BatchNormalization()(bl_right_route)
    bl_right_route = Activation('relu')(bl_right_route)
    # Right conv dw 3x3
    bl_right_route = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False)(bl_right_route)
    bl_right_route = BatchNormalization()(bl_right_route)
    bl_right_route = Activation('relu')(bl_right_route)
    # Right conv 1x1
    bl_right_route = Conv2D(filters=filts, kernel_size=1, strides=1, padding='same', data_format='channels_last', use_bias=False)(bl_right_route)
    bl_right_route = BatchNormalization()(bl_right_route)
    # Final concatenation
    bl_concat = Add()([bl_left_route, bl_right_route])
    bl_concat = Activation('relu')(bl_concat)
    return bl_concat

def D_stage(input_layer, filts, isConcat=False, concat_layer=None):
    bl = D_Block(input_layer, filts, isConcat, concat_layer)
    bl = UpSampling2D()(bl)
    bl = Dropout(0.2)(bl)
    return bl

def E_stage(input_layer, filts):
    bl = Conv2D(filters=filts, kernel_size=1, strides=1, padding='same', data_format='channels_last', use_bias=False)(input_layer)
    bl = BatchNormalization()(bl)
    bl = Activation('relu')(bl)
    bl = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(bl)
    return bl

# Data preprocess
print('\nRunning Data Preprocess...')
X_route = 'EG1800/input/'
Y_route = 'EG1800/mask/'
files = os.listdir(X_route)
dataCount = 0

# Actually available data
for file in files:
    dataCount += 1

dataSize = 224
X_chs = 3
X_data = np.empty((dataCount, dataSize, dataSize, X_chs), dtype=np.float16)
Y_mask = np.empty((dataCount, dataSize, dataSize), dtype=np.float16)
Y_boundary = np.empty((dataCount, dataSize, dataSize), dtype=np.float16)
count = 0

for i in range(0, 2632):
    img = X_route + '%05d' % (i + 1) + '.jpg'
    mask = Y_route + '%05d' % (i + 1) + '_mask.png'
    if not os.path.exists(img) or not os.path.exists(mask):
        continue
    imgMat = cv.imread(img)
    maskMat = cv.imread(mask)
    imgMat = cv.resize(imgMat, (dataSize, dataSize))
    maskMat = cv.resize(maskMat, (dataSize, dataSize))
    X_data[count] = imgMat / 255
    Y_mask[count] = cv.cvtColor(maskMat, cv.COLOR_BGR2GRAY) / 255
    maskMat_bin = cv.cvtColor(maskMat, cv.COLOR_BGR2GRAY)
    Y_boundary[count] = cv.Canny(maskMat_bin, 50, 255) / 255
    count += 1

Y_mask = tf.expand_dims(Y_mask, axis=-1)
Y_boundary = tf.expand_dims(Y_boundary, axis=-1)
split = int(dataCount * 0.8)
X_train, Y_mask_train, Y_boundary_train = X_data[:split], Y_mask[:split], Y_boundary[:split]
X_test, Y_mask_test, Y_boundary_test = X_data[split:], Y_mask[split:], Y_boundary[split:]
print('Preprocess complete.\nTraining start...')
# Data preprocess complete

main_input = Input((dataSize, dataSize, X_chs), name='main_input')
initlyr = Conv2D(filters=1, kernel_size=1, padding='same', data_format='channels_last', use_bias=False)(main_input)

enclyr_1 = E_stage(initlyr, 32)
enclyr_2 = E_stage(enclyr_1, 64)
enclyr_3 = E_stage(enclyr_2, 128) 
enclyr_4 = E_stage(enclyr_3, 256)
enclyr_5 = E_stage(enclyr_4, 512)

declyr_1 = D_stage(enclyr_5, 256)
declyr_2 = D_stage(declyr_1, 128, True, enclyr_4)
declyr_3 = D_stage(declyr_2, 64, True, enclyr_3)
declyr_4 = D_stage(declyr_3, 32, True, enclyr_2)
declyr_5 = D_stage(declyr_4, 16, True, enclyr_1)

mask = Conv2D(filters=1, kernel_size=1, use_bias=False, data_format='channels_last', activation='sigmoid')(declyr_5)
print(mask)
boundary = Conv2D(filters=1, kernel_size=1, use_bias=False, data_format='channels_last', activation='sigmoid')(declyr_5)
print(boundary)

model = Model(main_input, [mask, boundary], name='PortraitNet')
plot_model(model, to_file='structure.png')

adam = Adam(learning_rate=1e-3)

model.compile(
    optimizer=adam,
    loss=['binary_crossentropy', binary_focal_loss],
    loss_weights=[1., 0.1],
    metrics=[['accuracy'], ['accuracy']]    
)

history = model.fit(X_train, [Y_mask_train, Y_boundary_train], batch_size=16, epochs=25)
print(history.history.keys())

loss = model.evaluate(X_test, [Y_mask_test, Y_boundary_test])
print('Result: ', loss)

model.save('PortraitNet.h5')

plt.plot(history.history['conv2d_21_accuracy'])
plt.plot(history.history['conv2d_22_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Mask', 'Boundary'], loc='lower right')
plt.savefig('accu.png')

plt.plot(history.history['conv2d_21_loss'])
plt.plot(history.history['conv2d_22_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Mask', 'Boundary'], loc='upper right')
plt.savefig('loss.png')

# - loss: 0.2689 - conv2d_21_loss: 0.2682 - conv2d_22_loss: 0.0065 - conv2d_21_accuracy: 0.9140 - conv2d_22_accuracy: 0.9884
