import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import numpy as np
import keras
import tensorflow as tf
import h5py
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras_efficientnets import EfficientNetB3

#Fix seed to ensure experiment reproducibility
seed = 42
tf.set_random_seed(seed)
np.random.seed(seed)

saved_model_name = "model.h5"

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'data', 'train')
valid_dir = os.path.join(base_dir, 'data', 'valid')

batch_size = 16
input_dimensions = (300, 300, 3)
num_epochs = 1000000
patience = 100
num_classes = len(os.listdir(train_dir))
num_training_samples = sum([len(os.listdir(os.path.join(train_dir, cat_train_dir))) for cat_train_dir in os.listdir(train_dir)])
num_valid_samples = sum([len(os.listdir(os.path.join(valid_dir, cat_valid_dir))) for cat_valid_dir in os.listdir(valid_dir)])


conv_base = EfficientNetB3(input_dimensions, include_top=False, weights='imagenet')
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

train_datagen = ImageDataGenerator(rotation_range=20.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator()

# callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
model_checkpoint = ModelCheckpoint(saved_model_name, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [early_stop, reduce_lr, model_checkpoint]

# generators
train_generator = train_datagen.flow_from_directory(train_dir, input_dimensions[:-1], batch_size=batch_size,
                                                        class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir, input_dimensions[:-1], batch_size=batch_size,
                                                        class_mode='categorical')

# compile model
sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# finetune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=num_training_samples / batch_size,
    validation_data=valid_generator,
    validation_steps=num_valid_samples / batch_size,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=2)