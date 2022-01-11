import tensorflow_addons as tfa
from tensorflow import keras

from .base import BaseAutoEncoder
from .image_classifier import ImageClassifier
from .encoding_classifier import EncodingClassifier
from .autoencoder_classifier import AutoencoderClassifier
from ..losses import weighted_categorical_crossentropy, weighted_categorical_crossentropy2


class PrivacyAutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        z_dim=64,
        n_classes=2,
        encoder_weights_path=None,
        decoder_weights_path=None,
        autoencoder_weights_path=None,
        privacy_classifier_weights_path=None,
        utility_classifier_weights_path=None,
        # privacy_IMG_classifier_weights_path=None,
        # utility_IMG_classifier_weights_path=None,
        reconstruction_loss_weight=1.0,
        privacy_classifier_loss_weight=1.0,
        # privacy_IMG_classifier_loss_weight=1.0,
        utility_IMG_classifier_loss_weight=1.0,
        # encoding_classifier_loss_weight=1.0,
        utility_classifier_loss_weight=1.0,
        crossentropy_weights_privacy=[1e-5, 1.0],
        # crossentropy_weights_IMG_privacy=[1.0, 1e-2],
        # crossentropy_weights_IMG_utility=[1.0, 1e-2],
        crossentropy_weights_utility=[1e-5, 1.0],
        opt=keras.optimizers.Adam(lr=1e-3),
    ):

        self.input_shape = input_shape
        self.n_channels = input_shape[-1]
        self.z_dim = z_dim
        self.n_classes = n_classes

        # Load autoencoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        if encoder_weights_path is not None:
            self.encoder.load_weights(encoder_weights_path)

        if decoder_weights_path is not None:
            self.decoder.load_weights(decoder_weights_path)

        self.autoencoder = self.build_autoencoder()

        if autoencoder_weights_path is not None:
            self.autoencoder.load_weights(autoencoder_weights_path)

        # Build classifiers
        self.encoded_privacy_classifier = self.load_encoding_classifier(
            privacy_classifier_weights_path
        )
        self.encoded_utility_classifier = self.load_encoding_classifier(
            utility_classifier_weights_path
        )

        # # load img classifier
        # self.utility_IMG_classifier = self.load_classifier(
        #     utility_IMG_classifier_weights_path
        # )

        # Construct combined model with frozen classifiers attached
        self.model = self.combine_models2(
            opt,
            reconstruction_loss_weight,
            utility_classifier_loss_weight,
            privacy_classifier_loss_weight,
            # privacy_IMG_classifier_loss_weight,
            # utility_IMG_classifier_loss_weight,
            # encoding_classifier_loss_weight,
            crossentropy_weights_privacy,
            crossentropy_weights_utility,
            # crossentropy_weights_IMG_utility,
        )

        # self.model = self.combine_models3(
        #     opt,
        #     reconstruction_loss_weight,
        #     utility_classifier_loss_weight,
        #     privacy_classifier_loss_weight,
        #     privacy_IMG_classifier_loss_weight,
        #     # encoding_classifier_loss_weight,
        #     crossentropy_weights_privacy,
        #     crossentropy_weights_utility,
        #     crossentropy_weights_IMG_privacy,
        # )



    # def load_image_classifier(self, classifier_weights_path):
    #     """ Load the image classifier and its pretrained weights """
    #     model = ImageClassifier(
    #         input_shape=self.input_shape,
    #         z_dim=self.z_dim,
    #         n_classes=self.n_classes,
    #         classifier_weights_path=classifier_weights_path,
    #     )
    #     model = model.classifier
    #
    #     # Freeze the model
    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.trainable = False
    #
    #     return model

    def load_classifier(self, classifier_weights_path):
        """ Load the encoding classifier and its pretrained weights """
        model = ImageClassifier(
            input_shape=self.input_shape,
            z_dim=self.z_dim,
            n_classes=self.n_classes,
            classifier_weights_path=classifier_weights_path,
        )
        model = model.classifier

        # Freeze the model
        for layer in model.layers:
            layer.trainable = False
        model.trainable = False

        return model

    # def load_privacy_classifier(self, classifier_weights_path):
    #     """ Load the image classifier and its pretrained weights """
    #     model = ImageClassifier(
    #         input_shape=self.input_shape,
    #         z_dim=self.z_dim,
    #         n_classes=self.n_classes,
    #         classifier_weights_path=classifier_weights_path,
    #     )
    #     model = model.classifier
    #
    #     # Freeze the model
    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.trainable = False
    #
    #     return model
    #
    # def load_utility_classifier(self, classifier_weights_path):
    #     """ Load the encoding classifier and its pretrained weights """
    #     model = ImageClassifier(
    #         input_shape=self.input_shape,
    #         z_dim=self.z_dim,
    #         n_classes=self.n_classes,
    #         classifier_weights_path=classifier_weights_path,
    #     )
    #     model = model.classifier
    #
    #     # Freeze the model
    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.trainable = False
    #
    #     return model

    def load_encoding_classifier(self, classifier_weights_path):
        """ Load the encoding classifier and its pretrained weights """
        model = EncodingClassifier(
            input_shape=self.input_shape,
            z_dim=self.z_dim,
            n_classes=self.n_classes,
            classifier_weights_path=classifier_weights_path,
        )
        model = model.classifier
        # for layer in self.encoded_privacy_classifier.layers:
        #     layer.trainable = False
        # self.encoded_privacy_classif

        # Freeze the model
        for layer in model.layers:
            layer.trainable = False
        model.trainable = False

        return model

    # def load_utility_classifier(self, classifier_weights_path):
    #     """ Load the encoding classifier and its pretrained weights """
    #     model = EncodingClassifier(
    #         input_shape=self.input_shape,
    #         z_dim=self.z_dim,
    #         n_classes=self.n_classes,
    #         classifier_weights_path=classifier_weights_path,
    #     )
    #     model = model.classifier
    #
    #     # Freeze the model
    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.trainable = False
    #
    #     return model

    # def combine_models(
    #     self,
    #     opt,
    #     reconstruction_loss_weight,
    #     image_classifier_loss_weight,
    #     encoding_classifier_loss_weight,
    #     crossentropy_weights,
    # ):
    #     """ Combine the autoencoder with the pretrained and encoding classifiers """
    #
    #     inputs = keras.layers.Input(shape=self.input_shape)
    #     code = self.encoder(inputs)
    #
    #     # Encoding classifier output
    #     encoding_classifier_out = self.encoding_classifier(code)
    #     encoding_classifier_out = keras.layers.Lambda(
    #         lambda x: x, name="encoding_classifier_output"
    #     )(encoding_classifier_out)
    #
    #     # Freeze decoder
    #     for layer in self.decoder.layers:
    #         layer.trainable = False
    #     self.decoder.trainable = False
    #
    #     # Decoder output
    #     decoder_out = self.decoder(code)
    #     decoder_out = keras.layers.Lambda(lambda x: x, name="decoder_output")(
    #         decoder_out
    #     )
    #
    #     # Image Classifier output
    #     image_classifier_out = self.image_classifier(decoder_out)
    #     image_classifier_out = keras.layers.Lambda(
    #         lambda x: x, name="image_classifier_output"
    #     )(image_classifier_out)
    #
    #     # Compile combined model
    #     model = keras.models.Model(
    #         inputs=inputs,
    #         outputs=[decoder_out, encoding_classifier_out, image_classifier_out],
    #         name="PrivacyAutoencoderWithClassifiers",
    #     )
    #
    #     model.compile(
    #         opt,
    #         loss={
    #             "decoder_output": "mse",
    #             "encoding_classifier_output": weighted_categorical_crossentropy(
    #                 crossentropy_weights
    #             ),
    #             "image_classifier_output": weighted_categorical_crossentropy(
    #                 crossentropy_weights
    #             ),
    #         },
    #         loss_weights={
    #             "decoder_output": reconstruction_loss_weight,
    #             "encoding_classifier_output": encoding_classifier_loss_weight,
    #             "image_classifier_output": image_classifier_loss_weight,
    #         },
    #         metrics={
    #             "decoder_output": ["mse", "mae"],
    #             "encoding_classifier_output": [
    #                 "accuracy",
    #                 keras.metrics.AUC(),
    #                 keras.metrics.Precision(),
    #                 keras.metrics.Recall(),
    #                 tfa.metrics.F1Score(num_classes=self.n_classes),
    #             ],
    #             "image_classifier_output": [
    #                 "accuracy",
    #                 keras.metrics.AUC(),
    #                 keras.metrics.Precision(),
    #                 keras.metrics.Recall(),
    #                 tfa.metrics.F1Score(num_classes=self.n_classes),
    #             ],
    #         },
    #     )
    #
    #     print(model.summary())
    #     return model

    def combine_models2(
        self,
        opt,
        reconstruction_loss_weight,
        encoded_utility_classifier_loss_weight,
        encoded_privacy_classifier_loss_weight,
        crossentropy_weights_privacy,
        crossentropy_weights_utility,
    ):
        """ Combine the autoencoder with the pretrained and encoding classifiers """

        # # Freeze Encoder
        # for layer in self.encoder.layers:
        #     layer.trainable = False
        # self.encoder.trainable = False

        inputs = keras.layers.Input(shape=self.input_shape)
        code = self.encoder(inputs)

        # # Freeze decoder
        # for layer in self.decoder.layers:
        #     layer.trainable = False
        # self.decoder.trainable = False

        # # Decoder output
        # decoder_out = self.decoder(code)
        # decoder_out = keras.layers.Lambda(lambda z: z, name="decoder_output")(
        #     decoder_out
        # )

        # Freeze utility classifier
        for layer in self.encoded_utility_classifier.layers:
            layer.trainable = False
        self.encoded_utility_classifier.trainable = False

        # Utility classifier output
        encoded_utility_classifier_out = self.encoded_utility_classifier(code)
        encoded_utility_classifier_out = keras.layers.Lambda(
            lambda y: y, name="utility_classifier_output"
        )(encoded_utility_classifier_out)

        # Freeze Privacy classifier
        for layer in self.encoded_privacy_classifier.layers:
            layer.trainable = False
        self.encoded_privacy_classifier.trainable = False

        # Privacy Classifier output
        encoded_privacy_classifier_out = self.encoded_privacy_classifier(code)
        encoded_privacy_classifier_out = keras.layers.Lambda(
            lambda x: x, name="privacy_classifier_output"
        )(encoded_privacy_classifier_out)

        # Freeze decoder
        for layer in self.decoder.layers:
            layer.trainable = False
        self.decoder.trainable = False

        # Decoder output
        decoder_out = self.decoder(code)
        decoder_out = keras.layers.Lambda(lambda z: z, name="decoder_output")(
            decoder_out
        )

        # encoder_out2 = self.decoder(decoder_out)
        # encoder_out2 = keras.layers.Lambda(lambda x: x, name="encoder_output2")(
        #     encoder_out2
        # )

        # # Freeze decoder
        # for layer in self.encoder.layers:
        #     layer.trainable = False
        # self.encoder.trainable = False
        #
        # Lastupdate = self.encoder(decoder_out)

        # Compile combined model
        model = keras.models.Model(
            inputs=inputs,
            outputs=[decoder_out, encoded_utility_classifier_out, encoded_privacy_classifier_out],
            name="PrivacyAutoencoderWithClassifiers",
        )

        model.compile(
            opt,
            # loss={
            #     "decoder_output": "mae",
            #     "utility_classifier_output": weighted_categorical_crossentropy(
            #         crossentropy_weights_utility
            #     ),
            #     "privacy_classifier_output": weighted_categorical_crossentropy(
            #         crossentropy_weights_privacy
            #     ),
            #     # "encoder_output2": "mse",
            # },

            loss={
                "decoder_output": "mae",
                "utility_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights_utility
                ),
                "privacy_classifier_output": weighted_categorical_crossentropy2(
                    crossentropy_weights_privacy
                ),
                # "encoder_output2": "mse",
            },

            loss_weights={
                "decoder_output": reconstruction_loss_weight,
                "utility_classifier_output": encoded_utility_classifier_loss_weight,
                "privacy_classifier_output": encoded_privacy_classifier_loss_weight,
                # "encoder_output2": reconstruction_loss_weight,
            },
            metrics={
                "decoder_output": ["mse", "mae"],
                # "encoding_classifier_output": [
                #     "accuracy",
                #     keras.metrics.AUC(),
                #     keras.metrics.Precision(),
                #     keras.metrics.Recall(),
                #     tfa.metrics.F1Score(num_classes=self.n_classes),
                # ],
                "utility_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    # keras.metrics.Precision(),
                    # keras.metrics.Recall(),
                    # tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
                "privacy_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    # keras.metrics.Precision(),
                    # keras.metrics.Recall(),
                    # tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
            },
        )

        print(model.summary())
        return model

    def combine_models3(
        self,
        opt,
        reconstruction_loss_weight,
        encoded_utility_classifier_loss_weight,
        encoded_privacy_classifier_loss_weight,
        utility_IMG_classifier_loss_weight,
        crossentropy_weights_privacy,
        crossentropy_weights_utility,
        crossentropy_weights_IMG_utility,
    ):
        """ Combine the autoencoder with the pretrained and encoding classifiers """

        # # Freeze Encoder
        # for layer in self.encoder.layers:
        #     layer.trainable = False
        # self.encoder.trainable = False

        inputs = keras.layers.Input(shape=self.input_shape)
        code = self.encoder(inputs)

        # # Freeze decoder
        # for layer in self.decoder.layers:
        #     layer.trainable = False
        # self.decoder.trainable = False

        # # Decoder output
        # decoder_out = self.decoder(code)
        # decoder_out = keras.layers.Lambda(lambda z: z, name="decoder_output")(
        #     decoder_out
        # )

        # Freeze utility classifier
        for layer in self.encoded_utility_classifier.layers:
            layer.trainable = False
        self.encoded_utility_classifier.trainable = False

        # Utility classifier output
        encoded_utility_classifier_out = self.encoded_utility_classifier(code)
        encoded_utility_classifier_out = keras.layers.Lambda(
            lambda y: y, name="utility_classifier_output"
        )(encoded_utility_classifier_out)

        # Freeze Privacy classifier
        for layer in self.encoded_privacy_classifier.layers:
            layer.trainable = False
        self.encoded_privacy_classifier.trainable = False

        # Privacy Classifier output
        encoded_privacy_classifier_out = self.encoded_privacy_classifier(code)
        encoded_privacy_classifier_out = keras.layers.Lambda(
            lambda x: x, name="privacy_classifier_output"
        )(encoded_privacy_classifier_out)

        # Freeze decoder
        for layer in self.decoder.layers:
            layer.trainable = False
        self.decoder.trainable = False

        # Decoder output
        decoder_out = self.decoder(code)
        decoder_out = keras.layers.Lambda(lambda x: x, name="decoder_output")(
            decoder_out
        )

        # Freeze Utility IMG classifier
        for layer in self.utility_IMG_classifier.layers:
            layer.trainable = False
        self.utility_IMG_classifier.trainable = False

        # Privacy IMG Classifier output
        utility_IMG_classifier_out = self.utility_IMG_classifier(decoder_out)
        utility_IMG_classifier_out = keras.layers.Lambda(
            lambda y: y, name="utility_IMG_classifier_output"
        )(utility_IMG_classifier_out)

        # Compile combined model
        model = keras.models.Model(
            inputs=inputs,
            outputs=[decoder_out, encoded_utility_classifier_out, encoded_privacy_classifier_out,
                     utility_IMG_classifier_out],
            name="PrivacyAutoencoderWithClassifiers",
        )

        model.compile(
            opt,
            loss={
                "decoder_output": "mse",
                "utility_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights_utility
                ),
                "privacy_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights_privacy
                ),
                "utility_IMG_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights_IMG_utility
                ),
            },
            loss_weights={
                "decoder_output": reconstruction_loss_weight,
                "utility_classifier_output": encoded_utility_classifier_loss_weight,
                "privacy_classifier_output": encoded_privacy_classifier_loss_weight,
                "utility_IMG_classifier_output": utility_IMG_classifier_loss_weight,
            },
            metrics={
                "decoder_output": ["mse", "mae"],
                # "encoding_classifier_output": [
                #     "accuracy",
                #     keras.metrics.AUC(),
                #     keras.metrics.Precision(),
                #     keras.metrics.Recall(),
                #     tfa.metrics.F1Score(num_classes=self.n_classes),
                # ],
                "utility_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    # keras.metrics.Precision(),
                    # keras.metrics.Recall(),
                    # tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
                "privacy_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    # keras.metrics.Precision(),
                    # keras.metrics.Recall(),
                    # tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
                "utility_IMG_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    # keras.metrics.Precision(),
                    # keras.metrics.Recall(),
                    # tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
            },
        )

        print(model.summary())
        return model