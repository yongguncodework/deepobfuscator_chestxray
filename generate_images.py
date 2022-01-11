import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

# tf.config.threading.set_intra_op_parallelism_threads(16)
# tf.config.threading.set_inter_op_parallelism_threads(16)
# tf_config = tf.compat.v1.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# tf_config.log_device_placement = True
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


parser = argparse.ArgumentParser()
parser.add_argument('--nextstage', type=int, help='Next Stage')
parser.add_argument('--prevstage', type=int, help='Previous Stage')
args = parser.parse_args()

NEW = "privacy_stage{}".format(args.nextstage)
PREV = "privacy_stage{}".format(args.prevstage)

from privacy_encoder.models import AutoEncoder
from privacy_encoder.data import CelebA, Xray

# MODEL_DIR = "./models/"
MODEL_DIR = "./models/Xray6/"
# DATA_DIR = "./data/celeba/"
DATA_DIR = "./data/xray/"
# IMAGE_DIR = "./data/celeba/img_align_celeba_cropped/"

# MODEL_DIR = "./models/base/"
# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"

# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "./data/img_align_celeba/img_align_celeba/"
IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"

# RESULTS_DIR_RAW = "./data/output/raw_split12tr1_pos"
# RESULTS_DIR_RECON = "./data/output/reconstructed_split12tr1_pos"
# RESULTS_DIR_COMBINED = "./data/output/combined_split12tr1_pos"
# RESULTS_DIR_RAW = "./data/output/raw_split12tr1_neg"
# RESULTS_DIR_RECON = "./data/output/reconstructed_split12tr1_neg"
# RESULTS_DIR_COMBINED = "./data/output/combined_split12tr1_neg"
# RESULTS_DIR_RAW = "./data/output/rawXray6_neg"
# RESULTS_DIR_RECON = "./data/output/reconstructedXray6_neg"
# RESULTS_DIR_COMBINED = "./data/output/combinedXray6_neg"
RESULTS_DIR_RAW = "./data/output/rawXray6_all1"
RESULTS_DIR_RECON = "./data/output/reconstructedXray6_all1"
RESULTS_DIR_COMBINED = "./data/output/combinedXray6_all1"

# CLASSES = ["1", "3"]
CLASSES = ["0", "1"]
FEATURES = "PatientGender"
# FEATURES = "disease"

# CLASSES = [["0", "1"], ["1", "0"]]
# FEATURES = [str("PatientGender"), str("disease")]

INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
# ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stageNone/encoder.h5")
# DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stageNone/decoder.h5")

ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "encoder.h5")
DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "decoder.h5")
# ENCODER_WEIGHTS_PATH = None
# DECODER_WEIGHTS_PATH = None
# AUTOENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "autoencoder_Xray_base_final.h5")
# AUTOENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "autoencoder_Xray_base_final.h5")
AUTOENCODER_WEIGHTS_PATH = None
# AUTOENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "autoencoder_Xray.h5")
#

os.makedirs(RESULTS_DIR_RAW)
os.makedirs(RESULTS_DIR_RECON)
os.makedirs(RESULTS_DIR_COMBINED)

xray = Xray(image_folder=IMAGE_DIR, selected_features=FEATURES)
xray.attributes[FEATURES] = xray.attributes[FEATURES].replace({0: 1, 1: 0})

xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)
train_split = xray.split("training", drop_zero=False)
val_split = xray.split("validation", drop_zero=False)
test_split = xray.split("test", drop_zero=False)

def get_neg_class():
    # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = xray_data.split("test", drop_zero=False)
    # df = xray_data.split("training", drop_zero=False).iloc[:200]
    paths = df[df[FEATURES] == 0]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

def get_pos_class():
    # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = xray_data.split("test", drop_zero=False)
    # df = xray_data.split("training", drop_zero=False).iloc[:200]
    paths = df[df[FEATURES] == 1]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

def get_all_class():
    # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    xray_data = Xray(main_folder=DATA_DIR)
    df = xray_data.split("test", drop_zero=False)
    # df = xray_data.split("training", drop_zero=False).iloc[:200]
    paths = df["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    # random.shuffle(paths)
    return paths

from privacy_encoder.callbacks import ReconstructionByClass

neg_class_df = test_split[test_split["PatientGender"] != "0"]
pos_class_df = test_split[test_split["PatientGender"] == "0"]

# def get_class():
#     # celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
#     xray_data = Xray(main_folder=DATA_DIR, selected_features=[FEATURES])
#     df = xray_data.split("test", drop_zero=False)
#     paths = df[df[FEATURES]]["image_id"].tolist()
#     paths = [os.path.join(IMAGE_DIR, path) for path in paths]
#     random.shuffle(paths)
#     return paths

paths = get_all_class()
# paths = get_neg_class()
#
privacy_encoder = AutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
)

recon_sampler = ReconstructionByClass(
    encoder=AutoEncoder.build_encoder(ENCODER_WEIGHTS_PATH),
    decoder=AutoEncoder.build_decoder(DECODER_WEIGHTS_PATH),
    neg_class_df=neg_class_df,
    pos_class_df=pos_class_df,
    n_images=5,
    model_dir=MODEL_DIR,
    output_dir="./results/Xray6a/".format(NEW),
    image_dir=IMAGE_DIR,
)



for path in tqdm(paths):
    image = privacy_encoder.load_image(file_path=path)
    recon = privacy_encoder.predict(image)
    combined = np.concatenate((image, recon), axis=1)

    image, recon, combined = Image.fromarray(image), Image.fromarray(recon), Image.fromarray(combined)
    image.save(os.path.join(RESULTS_DIR_RAW, os.path.basename(path)))
    recon.save(os.path.join(RESULTS_DIR_RECON, os.path.basename(path)))
    combined.save(os.path.join(RESULTS_DIR_COMBINED, os.path.basename(path)))
