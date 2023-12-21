import tensorflow as tf
from tensorflow import keras

MODEL_NAME_IN='cardmodel_v2_40_0.913.h5'
MODEL_NAME_LITE='model/cardmodel.tflite'

model = keras.models.load_model(MODEL_NAME_IN)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(MODEL_NAME_LITE, 'wb') as f_out:
    f_out.write(tflite_model)

print('converting done..')