import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU, Add, MaxPooling2D,
                                     GlobalAveragePooling2D, Dense, Dropout, SpatialDropout2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from Utils import Utils


class DigitReader:

    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights('./public/aimodel/model_weights.weights.h5')

    def read_digits(self, img, cords, count):
        values = []

        for i in range(0, len(cords), count):
            string = ''
            for idx in range(count):
                digit = Utils.crop(img, cords[i + idx])
                digit, _ = self.predict(digit, self.model)
                string += str(digit)

            values.append(string)

        return values

    def preprocess_prediction_image(self, img, threshold=0.85):
        """
        img: (H, W, 3) uint8 [0,255] OR float32 [0,1]
        returns: (1, 28, 28, 1) float32
        digit = 1, background = 0
        """

        # 1. Convert to float32
        img = img.astype(np.float32)

        # 2. Normalize if needed
        if img.max() > 1.0:
            img /= 255.0

        # 3. RGB → Grayscale
        gray = (
                0.2989 * img[..., 0] +
                0.5870 * img[..., 1] +
                0.1140 * img[..., 2]
        )

        # 4. Resize to model input
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # 5. Binarize (digits dark → 1)
        binary = (gray < threshold).astype(np.float32)

        # 6. Add channel dim
        binary = binary[..., np.newaxis]

        # 7. Add batch dim
        binary = np.expand_dims(binary, axis=0)

        return binary

    #  Prediction Function
    def predict(self, img, model):
        img = self.preprocess_prediction_image(img)
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        return digit, np.max(prediction)

    def residual_block(self, x, filters, stride=1, l2_reg=5e-5):
        shortcut = x

        x = Conv2D(filters, 3, strides=stride, padding='same',
                   use_bias=False, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters, 3, padding='same',
                   use_bias=False, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)

        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv2D(filters, 1, strides=stride,
                              use_bias=False, kernel_regularizer=l2(l2_reg))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = LeakyReLU()(x)
        return x

    def build_model(self):
        inputs = Input(shape=(28, 28, 1))

        x = Conv2D(64, 3, padding='same', use_bias=False,
                   kernel_regularizer=l2(5e-5))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Residual stages
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)

        x = self.residual_block(x, 128, stride=2)
        x = self.residual_block(x, 128)

        x = self.residual_block(x, 256, stride=2)
        x = self.residual_block(x, 256)

        # Head
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)  # lowered from 0.3 - was compounding with augmentation + L2
        outputs = Dense(10, activation='softmax')(x)
        return Model(inputs, outputs)
