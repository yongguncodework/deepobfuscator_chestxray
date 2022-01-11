import os
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from privacy_encoder import models
from privacy_encoder.data import CelebA, multioutput_datagen, Xray, Xray2
from privacy_encoder.callbacks import ReconstructionByClass
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


parser = argparse.ArgumentParser()
parser.add_argument('--nextstage', type=int, help='Next Stage')
parser.add_argument('--prevstage', type=int, help='Previous Stage')
args = parser.parse_args()

NEW = "privacy_stage{}".format(args.nextstage)
PREV = "privacy_stage{}".format(args.prevstage)

# CLASSES = ["0", "1"]
# CLASSES = ["1", "3"]
# FEATURES = "PatientGender"
# FEATURES = "disease"

CLASSES = [["0", "1"], ["1", "0"]]
FEATURES = [str("PatientGender"), str("disease")]

N_CLASSES = 2

# MODEL_DIR = "./models/privacy_example/".format(NEW)
DATA_DIR = "/data/output6/"

MODEL_DIR = "./models/Xray6/"
# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"

# DATA_DIR = "/data/open-source/celeba/"
IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"

INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = "./models/Xray6/encoder.h5".format(PREV)
DECODER_WEIGHTS_PATH = "./models/Xray6/decoder.h5".format(PREV)
# ENCODER_WEIGHTS_PATH = None
# DECODER_WEIGHTS_PATH = None
AUTOENCODER_WEIGHTS_PATH = None
# AUTOENCODER_WEIGHTS_PATH = "./models/Xray3_1/autoencoder_Xray_base.h5".format(PREV)
UTILITY_CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/encoding_disease_classifier.h5".format(PREV)
PRIVACY_CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/encoding_gender_classifier.h5".format(PREV)
# PRIVACY_IMG_CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/privacy_classifier_gender.h5".format(PREV)
UTILITY_IMG_CLASSIFIER_WEIGHTS_PATH = "./models/Xray6/utility_classifier_gender.h5".format(PREV)
# UTILITY_CLASSIFIER_WEIGHTS_PATH = None
# PRIVACY_CLASSIFIER_WEIGHTS_PATH = None
EPOCHS = 30
BATCH_SIZE = 10

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# xray = Xray(image_folder=IMAGE_DIR, selected_features=[FEATURES])
xray = Xray(image_folder=IMAGE_DIR, selected_features=FEATURES)

# Flip labels to force autoencoder to fool classifiers
xray.attributes[FEATURES] = xray.attributes[FEATURES].replace({0: 1, 1: 0})

xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)

train_split = xray.split("training", drop_zero=False)
val_split = xray.split("validation", drop_zero=False)
test_split = xray.split("test", drop_zero=False)

# Random oversample smaller class (labels are flipped so glasses == '0')
high = len(train_split[train_split[FEATURES] != "0"][FEATURES])
low = len(train_split[train_split[FEATURES] == "0"][FEATURES])
n_samples = high - low
oversamples = train_split[train_split[FEATURES] == "0"].sample(n_samples, replace=True)
train_split = pd.concat([train_split, oversamples])

# Create datagens
datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=xray.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    # class_mode="categorical",
    class_mode="multi_output",
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
    # class_mode="categorical",
    class_mode="multi_output",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

# Build model
ae = models.PrivacyAutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    n_classes=N_CLASSES,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    utility_classifier_weights_path=UTILITY_CLASSIFIER_WEIGHTS_PATH,
    privacy_classifier_weights_path=PRIVACY_CLASSIFIER_WEIGHTS_PATH,
    # privacy_IMG_classifier_weights_path=PRIVACY_IMG_CLASSIFIER_WEIGHTS_PATH,
    # utility_IMG_classifier_weights_path=UTILITY_IMG_CLASSIFIER_WEIGHTS_PATH,
    reconstruction_loss_weight=1e+1,
    privacy_classifier_loss_weight=0.5*1e-1,
    utility_classifier_loss_weight=1e-3,
    # privacy_IMG_classifier_loss_weight=1e-1,
    # utility_IMG_classifier_loss_weight=1e-1,
    # encoding_classifier_loss_weight=1e-2,
    crossentropy_weights_privacy=[1, 1],
    # crossentropy_weights_IMG_utility=[1e-4, 1],
    crossentropy_weights_utility=[1e-4, 1],
    opt=keras.optimizers.Adam(lr=1e-3),
)
pprint(ae.model.input)
pprint(ae.model.output)

import pydot

# dot_img_file = 'model_1.png'
# plot_model(ae.model, to_file='model_1.png', show_shapes=True)
# plot_model(ae.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(log_dir="./logs/Xray6/privacy_autoencoder/".format(NEW))
'''
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/privacy_autoencoder_with_classifiers.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
'''
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Reconstruction callback
# neg_class_df = test_split[test_split[FEATURES] != "0"]
# pos_class_df = test_split[test_split[FEATURES] == "0"]

neg_class_df = test_split[test_split["PatientGender"] != "0"]
pos_class_df = test_split[test_split["PatientGender"] == "0"]

recon_sampler = ReconstructionByClass(
    encoder=ae.encoder,
    decoder=ae.decoder,
    neg_class_df=neg_class_df,
    pos_class_df=pos_class_df,
    n_images=5,
    model_dir=MODEL_DIR,
    output_dir="./results/Xray6/".format(NEW),
    image_dir=IMAGE_DIR,
)

chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/Xray6/autoencoder_Xray.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)

# Train
history = ae.model.fit(
    multioutput_datagen(train_generator),
    validation_data=multioutput_datagen(val_generator),
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(val_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, early_stop, chkpnt, recon_sampler],
    verbose=1,
    workers=1,
    max_queue_size=256,
)


print(history.history.keys())
# summarize history for accuracy

# a = ['loss', 'decoder_output_loss', 'encoding_classifier_output_loss', 'image_classifier_output_loss',
#      'decoder_output_mse', 'decoder_output_mae', 'encoding_classifier_output_accuracy',
#      'encoding_classifier_output_auc_2', 'encoding_classifier_output_precision_2',
#      'encoding_classifier_output_recall_2', 'encoding_classifier_output_f1_score',
#      'image_classifier_output_accuracy', 'image_classifier_output_auc_3',
#      'image_classifier_output_precision_3', 'image_classifier_output_recall_3', 'image_classifier_output_f1_score',
#      'val_loss', 'val_decoder_output_loss', 'val_encoding_classifier_output_loss', 'val_image_classifier_output_loss',
#      'val_decoder_output_mse', 'val_decoder_output_mae', 'val_encoding_classifier_output_accuracy',
#      'val_encoding_classifier_output_auc_2', 'val_encoding_classifier_output_precision_2',
#      'val_encoding_classifier_output_recall_2', 'val_encoding_classifier_output_f1_score',
#      'val_image_classifier_output_accuracy', 'val_image_classifier_output_auc_3',
#      'val_image_classifier_output_precision_3', 'val_image_classifier_output_recall_3', 'val_image_classifier_output_f1_score', 'lr']

plt.plot(history.history['loss'])
plt.plot(history.history['decoder_output_loss'])
plt.plot(history.history['utility_classifier_output_loss'])
plt.plot(history.history['privacy_classifier_output_loss'])
# plt.plot(history.history['utility_IMG_classifier_output_loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning curve')
# plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['loss', 'decoder_output_loss', 'utility_classifier_output_loss', 'privacy_classifier_output_loss',
            # 'utility_IMG_classifier_output_loss',
            'val_loss'], loc='upper left')
plt.savefig('Learning_Curve_Xray65.png')


plt.close()

# plt.plot(history.history['utility_classifier_output_accuracy'])
plt.plot(history.history['utility_classifier_output_auc_2'])
# plt.plot(history.history['privacy_classifier_output_accuracy'])
plt.plot(history.history['privacy_classifier_output_auc_3'])
# plt.plot(history.history['val_utility_classifier_output_accuracy'])
# plt.plot(history.history['utility_IMG_classifier_output_auc_5'])
# plt.plot(history.history['privacy_IMG_classifier_output_auc_5'])
plt.plot(history.history['val_utility_classifier_output_auc_2'])
# plt.plot(history.history['val_privacy_classifier_output_accuracy'])
plt.plot(history.history['val_privacy_classifier_output_auc_3'])
# plt.plot(history.history['val_privacy_IMG_classifier_output_auc_5'])
# plt.plot(history.history['val_utility_IMG_classifier_output_auc_5'])
plt.title('AUC Track')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['utility_classifier_output_auc', 'privacy_classifier_output_auc',
            # 'utility_IMG_classifier_output_auc',
            'val_utility_classifier_output_auc', 'val_privacy_classifier_output_auc',
            # 'val_utility_IMG_classifier_output_auc'
            ], loc='upper left')
plt.savefig('Learning_Curve_Xray65_auc.png')