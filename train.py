TRAIN_DIR='dataset/train'
VALIDATION_DIR='dataset/valid'
TEST_DIR='dataset/test'


EPOCHS_TO_TRAIN=20


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
import tensorflow.keras.applications
import tensorflow.keras.applications.xception 
import tensorflow.keras.applications.resnet
import tensorflow.keras.applications.vgg16
import tensorflow.keras.applications.inception_v3


chechpoint = keras.callbacks.ModelCheckpoint(
    'cardmodel_v2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


def make_model_v2(learning_rate=0.001,size1=128,size2=128,droprate=None):

    base_model=keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze convolutional layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False
    
    
    # Create a new model by adding custom dense layers on top of the pre-trained model
    if droprate is None:
        model3 = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(size1, activation="relu"),
            tf.keras.layers.Dense(size2, activation="relu"),
            tf.keras.layers.Dense(53, activation="softmax")
        ])
    else:
        model3 = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(size1, activation="relu"),
            tf.keras.layers.Dense(size2, activation="relu"),
            tf.keras.layers.Dropout(droprate),
            tf.keras.layers.Dense(53, activation="softmax")
        ])

        
        
    # Compile the model
    model3.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    
    return model3

def datasets_v2(aug=False, batch_size=32, 
                                    rotation_range=0,
                                    width_shift_range=0,
                                    height_shift_range=0,
                                    shear_range=0,
                                    zoom_range=0,
                                    horizontal_flip=False,
                                    vertical_flip=False, ):
    if aug:
        train_datagen_s = ImageDataGenerator(rescale=1./255,
                                                            rotation_range=rotation_range,
                                                            width_shift_range=width_shift_range,
                                                            height_shift_range=height_shift_range,
                                                            shear_range=shear_range,
                                                            zoom_range=zoom_range,
                                                            horizontal_flip=horizontal_flip,
                                                            vertical_flip=vertical_flip,
                                                            fill_mode='nearest')
    else:
        train_datagen_s = ImageDataGenerator(rescale=1./255)
        
    valid_datagen_s = ImageDataGenerator(rescale=1./255)
    test_datagen_s = ImageDataGenerator(rescale=1./255)

    
    train_data_s = train_datagen_s.flow_from_directory(TRAIN_DIR,
                                                       batch_size=batch_size,  
                                                       target_size=(224, 224), 
                                                       class_mode="categorical", 
                                                       seed=42)        
    
    valid_data_s = valid_datagen_s.flow_from_directory(VALIDATION_DIR,
                                                   batch_size=batch_size,
                                                   target_size=(224, 224),
                                                   class_mode="categorical",
                                                   seed=42)
    
    test_data_s = test_datagen_s.flow_from_directory(TEST_DIR,
                                                   batch_size=batch_size,
                                                   target_size=(224, 224),
                                                   class_mode="categorical",
                                                   seed=42)
    return train_data_s, valid_data_s, test_data_s




learning_rate = 0.001 
epochs=EPOCHS_TO_TRAIN
size1=128
size2=256

train_dataset, valid_dataset, test_dataset=datasets_v2(aug=False, )

model=make_model_v2(learning_rate=learning_rate, size1=size1, size2=size2,droprate=0.2)

print(model.summary())

history = model.fit(train_dataset,
                    epochs=epochs,
                    steps_per_epoch=len(train_dataset),
                    validation_data=valid_dataset,
                    validation_steps=len(valid_dataset),
                    callbacks=[chechpoint] )   

scores_best=history.history 


model.evaluate(test_dataset)
