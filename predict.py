import argparse
parser = argparse.ArgumentParser(description='produce predictions')
parser.add_argument('--custom_test', action='store_true', help='flag for testing custom data. refer to readme')
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import keras
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras_efficientnets import EfficientNetB3

model = keras.models.load_model("model_final.h5")
base_dir = os.getcwd()
if(args.custom_test == True):
    test_dir = os.path.join(base_dir, 'data', 'custom_test')
else:
    test_dir = os.path.join(base_dir, 'data', 'test')
images_dir = os.path.join(test_dir, 'images')
num_test_samples = len(os.listdir(images_dir))

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_dir, (300,300), batch_size=1,
                                                  class_mode=None, shuffle=False)
preds = model.predict_generator(test_generator, num_test_samples)

if(args.custom_test):
    labels_w_prob = open('custom_pred_labels_with_probabilities.txt', 'w')
    labels = open('custom_pred_labels.txt', 'w')
else:
    labels_w_prob = open('pred_labels_with_probabilities.txt', 'w')
    labels = open('pred_labels.txt', 'w')
for i in range(num_test_samples):
    prob = np.max(preds[i])
    class_id = np.argmax(preds[i])
    labels_w_prob.write('{} {}\n'.format(str(class_id + 1), str(prob)))
    labels.write('{}\n'.format(str(class_id + 1)))
labels_w_prob.close()
labels.close()