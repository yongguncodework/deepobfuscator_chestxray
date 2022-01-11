import tensorflow_addons as tfa
from tensorflow import keras

from .base import BaseAutoEncoder


class AutoencoderClassifier(BaseAutoEncoder):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        z_dim=64,
        n_classes=2,
        encoder_weights_path=None,
        decoder_weights_path=None,
        classifier_weights_path=None,
        # autoencoder_weights_path=None,
        opt=keras.optimizers.Adam(lr=1e-3),
    ):

        self.input_shape = input_shape
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = input_shape[-1]

        # Build autoencoder head
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.classifier = self.build_classifier(opt)

        if encoder_weights_path is not None:
            self.encoder.load_weights(encoder_weights_path)

        if decoder_weights_path is not None:
            self.decoder.load_weights(decoder_weights_path)

        # Build autoencoder

        self.autoencoder = self.build_autoencoder()

        # if autoencoder_weights_path is not None:
        #     self.autoencoder.load_weights(autoencoder_weights_path)

        # Build classifier
        # self.classifier = self.build_classifier(opt)



        if classifier_weights_path is not None:
            self.classifier.load_weights(classifier_weights_path)

        self.model = self.build_combined(opt)

    # def build_classifier(self):
    #     """ Build the encoding classifier """
    #     inputs = keras.layers.Input(shape=(self.input_shape,))
    #     x = keras.layers.Dense(units=32)(inputs)
    #     x = keras.layers.Activation("relu")(x)
    #     x = keras.layers.Dense(units=32)(x)
    #     x = keras.layers.Activation("relu")(x)
    #     x = keras.layers.Dense(units=self.n_classes)(x)
    #     outputs = keras.layers.Activation("softmax")(x)
    #     # model = keras.models.Model(inputs, outputs, name="Encoding_Classifier")
    #     model = keras.models.Model(inputs, outputs)
    #     print(model.summary())
    #     return model

    def build_classifier(self, opt):
        inputs = keras.layers.Input(shape=self.input_shape)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=32)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dense(units=self.n_classes)(x)
        outputs = keras.layers.Activation("softmax")(x)
        model = keras.models.Model(inputs, outputs, name="Image_Classifier")
        # print(model.summary())
        # model.compile(
        #     optimizer=opt,
        #     loss="categorical_crossentropy",
        #     metrics=[
        #         "accuracy",
        #         keras.metrics.AUC(),
        #         keras.metrics.Precision(),
        #         keras.metrics.Recall(),
        #         tfa.metrics.F1Score(num_classes=self.n_classes),
        #     ],
        # )

        return model

    def build_combined(self, opt):
        """ Combine Encoder ->  Classifier """

        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        self.encoder.trainable = False

        inputs = keras.layers.Input(shape=self.input_shape)
        encoder_out = self.encoder(inputs)

        # Freeze the decoder
        for layer in self.decoder.layers:
            layer.trainable = False
        self.decoder.trainable = False

        decoder_out = self.decoder(encoder_out)

        # Image Classifier output
        encoding_classifier_out = self.classifier(decoder_out)

        # Compile combined model
        model = keras.models.Model(
            inputs=inputs,
            outputs=encoding_classifier_out,
            name="AutoencoderClassifierWithClassifier",
        )

        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes=self.n_classes),
            ],
        )

        print(model.summary())
        return model