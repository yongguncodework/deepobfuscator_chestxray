import argparse
from pprint import pprint
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from privacy_encoder import models
from privacy_encoder.data import CelebA, Xray, Xray2
from privacy_encoder.callbacks import ModelSaver

import tempfile, sys, os
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from tensorflow import keras
import numpy



tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, help='Next Stage')
args = parser.parse_args()

NEW = "privacy_stage_prediction".format(args.stage)

# CLASSES = ["0", "1"]
CLASSES = ["1", "3"]
# FEATURES = "PatientGender"
FEATURES = "disease"
N_CLASSES = 2

# DATA_DIR = "/data/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"
MODEL_DIR = "./models/Xray_base/"
# IMAGE_DIR = "./data/img_align_celeba/img_align_celeba/"
IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"

INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
# ENCODER_WEIGHTS_PATH = "./models/prediction/encoder.h5".format(NEW)
# DECODER_WEIGHTS_PATH = "./models/prediction/decoder.h5".format(NEW)
# AUTOENCODER_WEIGHTS_PATH = None
CLASSIFIER_WEIGHTS_PATH = None
EPOCHS = 1
BATCH_SIZE = 10

# xray = Xray(image_folder=IMAGE_DIR)
# xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)
xray = Xray(image_folder=IMAGE_DIR, selected_features=[FEATURES])
xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)

xray2 = Xray2(image_folder=IMAGE_DIR, selected_features=[FEATURES])
xray2.attributes[FEATURES] = xray2.attributes[FEATURES].astype(str)

train_split = xray.split("training", drop_zero=False)
val_split = xray.split("validation", drop_zero=False)
# test_split = xray.split("test", drop_zero=False)
test_split = xray2.split("test", drop_zero=False)

datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=xray.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=xray.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_split,
    directory=xray.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

model = models.CelebAImageClassifier(
    input_shape=INPUT_SHAPE,
    # z_dim=Z_DIM,
    n_classes=N_CLASSES,
    # encoder_weights_path=ENCODER_WEIGHTS_PATH,
    # decoder_weights_path=DECODER_WEIGHTS_PATH,
    # autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
)

from matplotlib import image

deep_data_x = test_generator.filenames
deep_y = np.array(test_generator.classes)


q1 = test_generator.filenames[0:10]
q2 = test_generator.classes[0:10]
q3 = image.imread("./data/output/reconstructedXray_base/" + q1[0])
q4 =  np.expand_dims(np.array(image.imread("./data/output/reconstructedXray_base/" + q1[1])), axis=0)
deep_x = np.zeros([10,128,128,3])
for i in range(0, 10):
    # w1 = np.expand_dims(image.imread("./data/output/reconstructedXray_base/" + q1[0]) , axis = 0)
    w2 = np.expand_dims(np.array(image.imread("./data/output/reconstructed_add/" + deep_data_x[i])), axis=0)
    deep_x[i] = w2


a =2
# img3 = np.append(q3, q4, axis=0)
# q3 = image.imread("./media/eeglab/YG_Storage/CT_Xray/{}/images/" + q1[0])

pprint(model.model.input)
pprint(model.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="auc", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(log_dir="./logs/prediction/image_classifier_deepexplain/".format(NEW))
'''
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/image_classifier_with_autoencoder.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
'''

chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="models/Xray_base/image_classifier_deepexplain.h5",
    monitor="val_auc",
    save_best_only=True,
    save_weights_only=True,
)

# early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", patience=10)
# model_saver = ModelSaver(
#     classifier=model.classifier, path="./models/prediction/image_classifier.h5".format(NEW)
# )

# Compute class weights
# class_weights = class_weight.compute_class_weight(
#     "balanced", np.unique(train_generator.classes), train_generator.classes
# )

history = model.model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(test_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, chkpnt],
    # class_weight=class_weights,
    verbose=1,
    workers=16,
    max_queue_size=128,
)


from deepexplain.tensorflow import DeepExplain

with DeepExplain(session=tf.compat.v1.get_default_session()) as de:  # <-- init DeepExplain context
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = model.model.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs=model.model.layers[-2].output)
    target_tensor = fModel(input_tensor)

    xs = deep_x[0:10]
    ys = deep_y[0:10]

    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    # attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    # attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
    # attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
    # attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    # attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    # attributions_sv = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)

from DeepExplain.examples.utils import plot, plt

img_rows = 128
img_cols = 128

n_cols = 6
n_rows = int(len(attributions_gradin) / 2)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_gradin)):
    row, col = divmod(i, 2)
    plot(xs[i].reshape(img_rows, img_cols), cmap='Greys', axis=axes[row, col*3]).set_title('original')
    # plot(a1.reshape(img_rows, img_cols), cmap='Greys', axis=axes[row, col * 3 + 1]).set_title('original-feature')
    plot(a1.reshape(img_rows, img_cols), axis=axes[row,col*3+1]).set_title('saliency_D-G_abs')
    plot(a2.reshape(img_rows, img_cols), cmap='Greys', axis=axes[row,col*3+2]).set_title('saliency_Image-(G_abs)')
    # plot(a1.reshape(img_rows,img_cols), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input_gender')
    # plot(a2.reshape(img_rows,img_cols), xi = xs[i], axis=axes[row,col*3+2]).set_title('saliency_gender')
plt.savefig('./plots/testing_run.png')

# list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('model auc')
# plt.ylabel('auc')
# plt.xlabel('epoch')
# plt.legend(['train_auc', 'test_auc'], loc='upper left')
# plt.savefig('Learning_Curve_Xray_img_classifer_fordeepexplain.png')

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train_loss', 'test_loss'], loc='upper right')
# plt.savefig('Learning_Curve_loss_obfusgender_malepredictorig.png')

# numpy.savetxt("history_maletofemale.txt", history, delimiter=",")
# model.model.summary()
#
# preds = model.model.predict(val_generator, verbose=1)
#
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
#
# roc_y_test = Y_test
# roc_preds = preds
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(roc_y_test.ravel(), roc_preds.ravel())
# auc_keras = auc(fpr_keras, tpr_keras)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(auc_keras))
# np.save('./results/fpr_keras_age_CNN', fpr_keras)
# np.save('./results/tpr_keras_age_CNN', tpr_keras)
# np.save('./results/auc_keras_age_CNN', auc_keras)
#
# # plt.plots(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve for age')
# plt.legend(loc='best')
# plt.savefig('ROC_AUC_plot_age_CNN.png')