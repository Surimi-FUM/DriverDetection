"""
Dice係数の損失関数の定義
・Dice係数はF値とも呼ばれる
・検知したいものが検知されているかどうか評価したい場合に使用する
"""
import tensorflow as tf
import tensorflow.python.keras.backend as KB


class DiceLoss:
    def __init__(self, input_shape, class_num, ratios=None):
        self.__input_h = input_shape[0]
        self.__input_w = input_shape[1]
        self.__class_num = class_num
        self.__ratios = ratios
        if self.__ratios is None:
            self.__ratios = [3] * class_num

    def dice_coef_loss(self, y_true, y_pred):
        y_trues = self.__separate_by_class(y_true)
        y_preds = self.__separate_by_class(y_pred)

        losses = []
        for y_t, y_p, ratio in zip(y_trues, y_preds, self.__ratios):
            losses.append((1 - self.__dice_coef(y_t, y_p)) * ratio)

        return tf.reduce_sum(tf.stack(losses))

    def __separate_by_class(self, y):
        y_res = tf.reshape(y, (-1, self.__input_h, self.__input_w, self.__class_num))
        ys = tf.unstack(y_res, axis=3)
        return ys

    # Dice係数の計算
    @tf.function
    def __dice_coef(self, y_true, y_pred):
        y_true = KB.flatten(y_true)
        y_pred = KB.flatten(y_pred)
        intersection = KB.sum(y_true * y_pred)
        denominator = KB.sum(y_true) + KB.sum(y_pred)
        if denominator == 0:
            return 1.0
        if intersection == 0:
            return 1 / (denominator + 1)
        return (2.0 * intersection) / denominator
