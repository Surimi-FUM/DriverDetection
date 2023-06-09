"""
Unetの定義
・バッチ正規化層などの追加
・tensorflowのkerasもあるが、1つでもそのライブラリがあると、学習時に使用できない関数が出るので注意
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation
from keras.layers import BatchNormalization


def UNet_with_batchnorm(input_shape, class_num):
    input_layer = Input(shape=input_shape)
    conv11 = Conv2D(64, (3, 3), padding='same')(input_layer)
    bn11 = BatchNormalization()(conv11)
    act11 = Activation(activation='relu')(bn11)
    conv12 = Conv2D(64, (3, 3), padding='same')(act11)
    bn12 = BatchNormalization()(conv12)
    act12 = Activation(activation='relu')(bn12)
    pool1 = MaxPooling2D()(act12)

    conv21 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn21 = BatchNormalization()(conv21)
    act21 = Activation(activation='relu')(bn21)
    conv22 = Conv2D(128, (3, 3), padding='same')(act21)
    bn22 = BatchNormalization()(conv22)
    act22 = Activation(activation='relu')(bn22)
    pool2 = MaxPooling2D()(act22)

    conv31 = Conv2D(512, (3, 3), padding='same')(pool2)
    bn31 = BatchNormalization()(conv31)
    act31 = Activation(activation='relu')(bn31)
    conv32 = Conv2D(512, (3, 3), padding='same')(act31)
    bn32 = BatchNormalization()(conv32)
    act32 = Activation(activation='relu')(bn32)

    up1 = UpSampling2D()(act32)
    concat1 = Concatenate()([up1, conv22])
    conv41 = Conv2D(128, (3, 3), padding='same')(concat1)
    bn41 = BatchNormalization()(conv41)
    act41 = Activation(activation='relu')(bn41)
    conv42 = Conv2D(128, (3, 3), padding='same')(act41)
    bn42 = BatchNormalization()(conv42)
    act42 = Activation(activation='relu')(bn42)

    up2 = UpSampling2D()(act42)
    concat2 = Concatenate()([up2, conv12])
    conv51 = Conv2D(64, (3, 3), padding='same')(concat2)
    bn51 = BatchNormalization()(conv51)
    act51 = Activation(activation='relu')(bn51)
    conv52 = Conv2D(64, (3, 3), padding='same')(act51)
    bn52 = BatchNormalization()(conv52)
    act52 = Activation(activation='relu')(bn52)

    output_layer = Conv2D(class_num, (1, 1), activation='sigmoid', padding='same')(act52)

    return Model(input_layer, output_layer)
