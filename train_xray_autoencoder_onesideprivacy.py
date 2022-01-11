import os
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

from privacy_encoder import models
from privacy_encoder.data import CelebA, Xray
from privacy_encoder.callbacks import Reconstruction

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

# MODEL_DIR = "./models/base/"
MODEL_DIR = "./models/Xray/"
# DATA_DIR = "/data/open-source/celeba/"
# IMAGE_DIR = "img_align_celeba_cropped/"

# DATA_DIR = "/data/open-source/celeba/"
IMAGE_DIR = "/media/eeglab/YG_Storage/CT_Xray/images/"


INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512 #from 512
ENCODER_WEIGHTS_PATH = None
DECODER_WEIGHTS_PATH = None
AUTOENCODER_WEIGHTS_PATH = None
EPOCHS = 30
BATCH_SIZE = 16 #64

# CLASSES = "0"
# # CLASSES = ["1", "3"]
# FEATURES = "PatientGender"
# # FEATURES = "disease"

# CLASSES = ["1", "3"]
CLASSES = ["0", "1"]
FEATURES = "PatientGender"
# FEATURES = "disease"

# import os
# path = "./TEST"
#
# fname = []
# for root,d_names,f_names in os.walk(path):
# 	for f in f_names:
# 		fname.append(os.path.join(root, f))
#
# print("fname = %s" %fname)

# xray = Xray(image_folder=IMAGE_DIR)


xray = Xray(image_folder=IMAGE_DIR, selected_features=[FEATURES])
xray.attributes[FEATURES] = xray.attributes[FEATURES].astype(str)

train_split = xray.split("training", drop_zero=False)
val_split = xray.split("validation", drop_zero=False)
test_split = xray.split("test", drop_zero=False)


datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=xray.images_folder,
    x_col="image_id",
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="input",
    shuffle=True,
    color_mode="rgb",
    # color_mode="grayscale",
    interpolation="bilinear",
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=xray.images_folder,
    x_col="image_id",
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="input",
    shuffle=True,
    color_mode="rgb",
    # color_mode="grayscale",
    interpolation="bilinear",
)

ae = models.AutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
)
pprint(ae.model.input)
pprint(ae.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(log_dir="./logs/Xray/autoencoder_Xray_both/")
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/Xray/autoencoder_Xray_both.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
recon_sampler = Reconstruction(
    encoder=ae.encoder,
    decoder=ae.decoder,
    train_df=train_split,
    test_df=test_split,
    n_images=5,
    model_dir=MODEL_DIR,
    output_dir="./results/Xray_both/",
    # image_dir=DATA_DIR + IMAGE_DIR,
    image_dir=IMAGE_DIR,
)

ae.model.compile(
    # optimizer=keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mse", "mae"]
    optimizer=keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mse", "mae"]
    # optimizer=keras.optimizers.Adam()
)

history = ae.model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(val_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, chkpnt, early_stop, recon_sampler],
    verbose=1,
    workers=16,
    max_queue_size=128,
)
import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('model auc')
# plt.ylabel('auc')
# plt.xlabel('epoch')
# plt.legend(['train_auc', 'test_auc'], loc='upper left')
# plt.savefig('Learning_Curve_d_test2.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.plot(history.history['categorical_crossentropy'])
# plt.plot(history.history['val_categorical_crossentropy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train_loss', 'val_loss', 'categorical_crossentropy', 'val_categorical_crossentropy'], loc='upper right')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.savefig('Learning_Curve_loss_deeepobfuscator_both.png')