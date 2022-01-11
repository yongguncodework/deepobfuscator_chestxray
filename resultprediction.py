import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import argparse
import tensorflow as tf
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

from privacy_encoder.models import AutoEncoder
from privacy_encoder.data import CelebA, Xray

MODEL_DIR2 = "./models/"
# MODEL_DIR = "./models/Xray/"
MODEL_DIR = "./models/prediction/"
# DATA_DIR = "./data/celeba/"
DATA_DIR = "./data/xray/"
# IMAGE_DIR = "./data/celeba/img_align_celeba_cropped/"

# MODEL_DIR = "./models/base/"
# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"

# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "./data/img_align_celeba/img_align_celeba/"
# IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"

# RESULTS_DIR_RAW = "./data/output/raw"
# RESULTS_DIR_RECON = "./data/output/reconstructed"
# RESULTS_DIR_COMBINED = "./data/output/combined"
# RESULTS_DIR_RAW = "./data/output/raw2"
# RESULTS_DIR_RECON = "./data/output/reconstructed2"
# RESULTS_DIR_COMBINED = "./data/output/combined2"

IMAGE_DIR = "./data/output/reconstructed7/"
# IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"

CLASSES = ["0", "1"]
FEATURES = "PatientGender"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
# ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stageNone/encoder.h5")
# DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stageNone/decoder.h5")

# ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "encoder.h5")
# DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "decoder.h5")

CLASSIFICATION_WEIGHTS_PATH = os.path.join(MODEL_DIR2, "raw_image_classifier.h5")
AUTOENCODER_WEIGHTS_PATH = None


# os.makedirs(RESULTS_DIR_RAW)
# os.makedirs(RESULTS_DIR_RECON)
# os.makedirs(RESULTS_DIR_COMBINED)

# def get_neg_class():
#     # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
#     xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
#     df = xray_data.split("test", drop_zero=False)
#     paths = df[df[FEATURES] == 0]["image_id"].tolist()
#     paths = [os.path.join(IMAGE_DIR, path) for path in paths]
#     random.shuffle(paths)
#     return paths
#
# def get_pos_class():
#     # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
#     xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
#     df = xray_data.split("test", drop_zero=False)
#     paths = df[df[FEATURES] == 1]["image_id"].tolist()
#     paths = [os.path.join(IMAGE_DIR, path) for path in paths]
#     random.shuffle(paths)
#     return paths

# paths = get_pos_class()

# privacy_encoder = AutoEncoder(
#     input_shape=INPUT_SHAPE,
#     z_dim=Z_DIM,
#     encoder_weights_path=ENCODER_WEIGHTS_PATH,
#     decoder_weights_path=DECODER_WEIGHTS_PATH,
#     autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
# )
from privacy_encoder import models

CLASSES = ["0", "1"]
FEATURES = "PatientGender"
# FEATURES = "disease"
N_CLASSES = 2
BATCH_SIZE = 10

xray = Xray(image_folder=IMAGE_DIR, selected_features=[FEATURES])
xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)
train_split = xray.split("training", drop_zero=False)
val_split = xray.split("validation", drop_zero=False)
test_split = xray.split("test", drop_zero=False)


datagen = ImageDataGenerator(rescale=1.0 / 255)
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

classifier = models.CelebAImageClassifier(
    input_shape=INPUT_SHAPE,
    # z_dim=Z_DIM,
    n_classes=N_CLASSES,
    # encoder_weights_path=ENCODER_WEIGHTS_PATH,
    # decoder_weights_path=DECODER_WEIGHTS_PATH,
    # autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    classifier_weights_path=CLASSIFICATION_WEIGHTS_PATH,
)

def get_neg_class():
    # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = xray_data.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 0]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

def get_pos_class():
    # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = xray_data.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 1]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

# history = classifier.model.predict(test_generator)
paths = get_neg_class()

for path in tqdm(paths):
    recon = classifier.model.predict(test_generator)

# answer1  = classifier.predict_autoencoder(test_generator)

aa = 11

