import cv2 as cv
import numpy as np
import time
import tensorflow as tf
from keras import backend
from keras.models import Model, load_model

def binary_focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    eps = backend.epsilon()
    pt_1 = backend.clip(pt_1, eps, 1. - eps)
    pt_0 = backend.clip(pt_0, eps, 1. - eps)
    return -(backend.mean(alpha * backend.pow(1. - pt_1, gamma) * backend.log(pt_1), axis=-1) \
        + backend.mean((1 - alpha) * backend.pow(pt_0, gamma) * backend.log(1. - pt_0), axis=-1))

model = load_model('PortraitNet.h5', custom_objects={'binary_focal_loss': binary_focal_loss})

for i in range(1, 7):
    src = cv.imread('infer_%d.jpg' % i)
    src = cv.resize(src, (224, 224)) / 255
    cv.imwrite('src%d.png' % i, src * 255)
    mask, boundary = model.predict(np.reshape(src, (1, 224, 224, 3)))
    cv.imwrite('src%d_mask.png' % i, mask[0] * 255)
    cv.imwrite('src%d_bound.png' % i, boundary[0] * 255)



