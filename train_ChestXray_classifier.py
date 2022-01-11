import os
import argparse
from pprint import pprint
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from privacy_encoder import models
from privacy_encoder.data import CelebA
from privacy_encoder.callbacks import ModelSaver

from utils2 import get_celeba

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))



from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys
import os
import pickle
from os.path import abspath, dirname

from sklearn.model_selection import train_test_split
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

np.random.seed(1337)  # for reproducibility


img1 = np.array(np.load("/home/eeglab/PycharmProjects/YG/Lung_data/CT_Xray/Chest_Xray_15disease_images.npy"))
img2 = img1[:, ::8, ::8]
# img2  = img1

# data_file_name = 'Chest_Xray_Label.mat'
# data_file_name2 = 'Chest_Xray_Label_sci.mat'
data_file_name = 'Chest_Xray_15disease_Label.mat'
data_file_name2 = 'Chest_Xray_15disease_Label_sci.mat'
DATA_FOLDER_PATH = '/media/eeglab/YG_Storage/CT_Xray'
label_path = '/media/eeglab/YG_Storage/'
FILE_PATH = label_path + '/' + data_file_name
FILE_PATH2 = label_path + '/' + data_file_name2

mat = loadmat(FILE_PATH2)

age = np.array(mat['age'])
disease = np.array(mat['id'])
gender = np.array(mat['gender'])
image = np.array(mat['image'])
image_sci = (mat['image'])

data_augmentation = True
batch_size = 16
num_classes = 2
epochs = 100
# input image dimensions
img_rows, img_cols = 128, 128
# The CIFAR10 images are RGB.
img_channels = 1

nb_classes = 2

# Cardiomegaly and Edema

# Cardiomegaly_Edema = np.append(disease[1000:1999], disease[3000:3999])
Cardiomegaly_Edema_disease = np.append(np.zeros(1000), np.ones(1000), axis=0)
# Cardiomegaly_Edema_disease = np.append(disease[1000:2000], disease[3000:4000], axis=0)
Cardiomegaly_Edema_gender = np.append(gender[1000:2000], gender[3000:4000], axis=0)
Cardiomegaly_Edema = np.append(Cardiomegaly_Edema_disease, Cardiomegaly_Edema_gender)
ce = np.reshape(Cardiomegaly_Edema, [2, 2000])
img3 = np.append(img2[1000:2000, :, :], img2[3000:4000, :, :], axis=0)
# img3 = np.reshape(img3, [2000, 32 * 32])
# img3 = np.reshape(img3, [2000, 64 * 64])
# The data, shuffled and split between train and test sets:
# X_train, X_test, y_train, y_test = train_test_split(img3, Cardiomegaly_Edema_disease, test_size=0.20,
#                                                     random_state=42)
X_train, X_test, y_train, y_test = train_test_split(img3, Cardiomegaly_Edema_disease, test_size=0.20,
                                                    random_state=42)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)



input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = (X_train - 0.5) * 2
X_test = (X_test - 0.5) * 2
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


CLASSES = ['0', '1']
# FEATURES = 'Eyeglasses'
N_CLASSES = 1

import torchvision
# celeba = torchvision.datasets.CelebA("img_align_celeba/img_align_celeba/")
# celeba = torchvision.datasets.ImageFolder("img_align_celeba/img_align_celeba/")

# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"
IMAGE_DIR = "data/img_align_celeba/img_align_celeba/"
INPUT_SHAPE = [128, 128, 1]
CLASSIFIER_WEIGHTS_PATH = None
EPOCHS = 10
BATCH_SIZE = 64

# celeba = CelebA(image_folder=IMAGE_DIR, selected_features=[FEATURES])
# celeba.attributes[FEATURES] = celeba.attributes[FEATURES].astype(str)
# train_split = celeba.split("training", drop_zero=False)
# val_split = celeba.split("validation", drop_zero=False)

train_split = (X_train, y_train)
val_split = (X_test, y_test)

# train_split = X_train.split("training", drop_zero=False)
# val_split = X_test.split("validation", drop_zero=False)

# datagen = ImageDataGenerator(rescale=1.0 / 255)
# train_generator = datagen.flow_from_dataframe(
#     dataframe=X_train,
#     target_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     # y_col=y_train,
#     classes=y_train,
#     class_mode="categorical",
#     shuffle=True,
#     color_mode="grayscale",
#     interpolation="bilinear"
# )
#
# val_generator = datagen.flow_from_dataframe(
#     dataframe=val_split,
#     target_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     y_col=y_train,
#     classes='disease',
#     class_mode="categorical",
#     shuffle=True,
#     color_mode="grayscale",
#     interpolation="bilinear",
# )

model = models.CTImageClassifier(
    input_shape=INPUT_SHAPE,
    n_classes=N_CLASSES,
    classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
)

pprint(model.model.input)
pprint(model.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.getcwd(), "logs/image_classifier/"),
    profile_batch=100000000
)
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="models/raw_image_classifier.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(y_train), y_train
)
print("Class Weights:\n", class_weights)
#
# model.model.fit(
#     x=train_split,
#     y=y_train,
#     validation_data=val_split,
#     epochs=EPOCHS,
#     steps_per_epoch=len(train_split) // BATCH_SIZE,
#     validation_steps=len(val_split) // BATCH_SIZE,
#     callbacks=[reduce_lr, tb, early_stop, chkpnt],
#     class_weight=class_weights,
#     verbose=1,
#     workers=8,
#     max_queue_size=64,
# )
a = len(X_train) // BATCH_SIZE
model.model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    validation_data=val_split,
    epochs=EPOCHS,
    # class_weight=class_weights,
    # batch_size=10,
    # validation_batch_size= 10,
    # steps_per_epoch=len(X_train) // BATCH_SIZE,
    # validation_steps=len(X_test) // BATCH_SIZE,
    # steps_per_epoch=100,
    # validation_steps=100,

    callbacks=[reduce_lr, tb, early_stop, chkpnt],
    verbose=1,
    workers=8,
    max_queue_size=64
)

# pred = model.model.predict(X_test, verbose=0)
# roc_y_test = y_test
# roc_preds = pred
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(roc_y_test.ravel(), roc_preds.ravel())
# auc_keras = auc(fpr_keras, tpr_keras)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(auc_keras))
# # np.save('./results/keras_2diseaseypreds_cadio_edema_gender2disease_pred_subgenderfeature', roc_preds)
# # np.save('./results/fpr_keras_2disease_cadio_edema_gender2disease_pred_subgenderfeature', fpr_keras)
# # np.save('./results/tpr_keras_2disease_cadio_edema_gender2disease_pred_subgenderfeature', tpr_keras)
# # np.save('./results/auc_keras_2disease_cadio_edema_gender2disease_pred_subgenderfeature', auc_keras)
#
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve for disease model disease(subgenderfeature) Prediction')
# plt.legend(loc='best')
# plt.savefig('./plots/ROC_AUC_plot_2diseaselabel.png')
