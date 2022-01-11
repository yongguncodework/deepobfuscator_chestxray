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

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, help='Next Stage')
args = parser.parse_args()

NEW = "privacy_stage{}".format(args.stage)

CLASSES = ["1", "0"]
# CLASSES = ["1", "3"]
FEATURES = "PatientGender"
# FEATURES = "disease"
N_CLASSES = 2

MODEL_DIR = "./models/Xray6/"
# IMAGE_DIR = "./data/img_align_celeba/img_align_celeba/"
IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
# ENCODER_WEIGHTS_PATH = "./models/Xray6/encoder.h5".format(NEW)
CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/privacy_classifier_gender.h5".format(NEW)
# CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/utility_classifier_gender.h5".format(NEW)

# CLASSIFIER_WEIGHTS_PATH = None
EPOCHS = 1
BATCH_SIZE = 10

# celeba = CelebA(image_folder=IMAGE_DIR, selected_features=[FEATURES])
# celeba.attributes[FEATURES] = celeba.attributes[FEATURES].astype(str)
# train_split = celeba.split("training", drop_zero=False)
# val_split = celeba.split("validation", drop_zero=False)

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
    shuffle=False,
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
    shuffle=False,
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
    shuffle=False,
    color_mode="rgb",
    interpolation="bilinear",
)

# model = models.EncodingClassifier(
#     input_shape=INPUT_SHAPE,
#     z_dim=Z_DIM,
#     n_classes=N_CLASSES,
#     encoder_weights_path=ENCODER_WEIGHTS_PATH,
#     classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
# )

model = models.CelebAImageClassifier(
    input_shape=INPUT_SHAPE,
    # z_dim=Z_DIM,
    n_classes=N_CLASSES,
    # encoder_weights_path=ENCODER_WEIGHTS_PATH,
    # decoder_weights_path=DECODER_WEIGHTS_PATH,
    # autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
)

pprint(model.model.input)
pprint(model.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
# tb = keras.callbacks.TensorBoard(log_dir="./logs/Xray6/encoding_gender_classifier/".format(NEW))
'''
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/encoding_classifier_with_encoder.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
'''
# early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
# model_saver = ModelSaver(
#     classifier=model.classifier, path="./models/Xray6/encoding_gender_classifier.h5".format(NEW)
# )

# Compute class weights
# class_weights = class_weight.compute_class_weight(
#     "balanced", np.unique(train_generator.classes), train_generator.classes
# )

STEP_SIZE_TEST=len(test_split) // BATCH_SIZE

# history = model.model.evaluate_generator(generator=val_generator)

# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
# test_generator.reset()

# test_generator.reset()

pred= model.model.predict_generator(test_generator,
# steps=400,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
import pandas as pd

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# filenames=test_generator.filenames
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})
# results.to_csv("results.csv",index=False)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc

# AUC plots
# pred = xray_model.predict(X_test, verbose=0)
# roc_y_test = y_test
# test_generator.classes
roc_preds = pred[:, 1]

true_array2 = test_generator.classes

true_array = keras.utils.to_categorical(true_array2, 2)
pred_array = np.array(predictions, dtype='int64')
target_names = ['0', '1']
# accuracy = accuracy_score(true_array, pred_array)

# print(classification_report(true_array2, pred_array, target_names=target_names))
# print("Accuracy is :", accuracy)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(true_array2, roc_preds)
auc_keras = auc(fpr_keras, tpr_keras)
#
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(auc_keras))
# # np.save('./results/keras_2diseaseypreds_cadio_edema_gender2disease_pred', roc_preds)
# # np.save('./results/fpr_keras_2disease_cadio_edema_gender2disease_pred', fpr_keras)
# # np.save('./results/tpr_keras_2disease_cadio_edema_gender2disease_pred', tpr_keras)
# # np.save('./results/auc_keras_2disease_cadio_edema_gender2disease_pred', auc_keras)
#
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for gender model disease Prediction')
plt.legend(loc='best')
# plt.savefig('./plots/ROC_AUC_plot_2diseaselabels_gender2disease_obfus_d1.png')
plt.savefig('./plots/ROC_AUC_plot_2diseaselabels_gender2disease_obfus_g1.png')

# history = model.model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS,
#     steps_per_epoch=len(train_split) // BATCH_SIZE,
#     validation_steps=len(val_split) // BATCH_SIZE,
#     callbacks=[reduce_lr],
#     # class_weight=class_weights,
#     verbose=1,
#     workers=8,
#     max_queue_size=64,
# )

# score = model.model.predict(train_generator, val_generator, verbose=0)



# print(history.history.keys())
# print("%s: %.2f%%" % (model.model.metrics_names[1], score[1]*100))
# summarize history for accuracy
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('model auc')
# plt.ylabel('auc')
# plt.xlabel('epoch')
# plt.legend(['train_auc', 'test_auc'], loc='upper left')
# plt.savefig('test_output_gender.png')
# plt.savefig('test_output_disease.png')