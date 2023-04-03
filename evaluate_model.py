"""
学習済みモデルの評価を行う
・学習に使用した教師データを使用する
・MeanIoUの計算と可視化
IoU = true_positives / (true_positives + false_positives + false_negatives)
"""
import time
import os
import matplotlib.pyplot as plt
from generaters import SegmentationDataGenerator
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tools import DiceLoss


# マスク画像がOne-Hot形式の場合はこちらを使う
def calc_OneHotMeanIoU(true_masks, pred_masks, num_classes=4, data_name="Train"):
    m = tf.keras.metrics.OneHotMeanIoU(num_classes=num_classes)
    iou = 0
    count = 0
    for t_mask, p_mask in zip(true_masks, pred_masks):
        count += 1
        m.update_state(t_mask, p_mask)
        iou += m.result().numpy()
        print(count, data_name + ' [OneHotMeanIoU]:', m.result().numpy())
    print('All [OneHotMeanIoU]:', iou / len(true_masks))


# マスク画像が普通ならこちら
def calc_MeanIoU(true_masks, pred_masks, num_classes, data_name="Train"):
    iou = 0
    count = 0
    m = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    for t_mask, p_mask in zip(true_masks, pred_masks):
        count += 1
        m.update_state(t_mask, p_mask)
        iou += m.result().numpy()
        print(count, data_name + ' [MeanIoU]:', m.result().numpy(),
              ", Class : True=", np.unique(t_mask), "Pred=", np.unique(p_mask))
    print('All [MeanIoU]:', iou / len(true_masks))


def main():
    """データ準備"""
    IDs = {'person': 15, 'bike': 14, 'car': 7, 'bicycle': 2}
    class_ids = [IDs['person']]
    target_base_Dir = os.path.join('.', 'datasets/VOC2012/target_data')
    class_base_Dir = os.path.join(target_base_Dir, 'person_')
    # trainはデータ数が多いのでvalidationデータで評価する
    DATA_NAME = 'validation'
    val_imgs_Dir = os.path.join(class_base_Dir, DATA_NAME, 'img')
    val_segs_Dir = os.path.join(class_base_Dir, DATA_NAME, 'seg')
    val_generater = SegmentationDataGenerator(input_dir=val_imgs_Dir, mask_dir=val_segs_Dir,
                                              image_shape=(128, 128), target_class_ids=class_ids)

    """モデル準備"""
    input_shape = (128, 128, 3)
    class_num = len(class_ids) + 1
    MODELNAME = 'unetBNDL_P'
    model_Dir = os.path.join('.', 'results', MODELNAME, 'learned_model_best')
    dl_loss = DiceLoss(input_shape=input_shape, class_num=class_num).dice_coef_loss
    # モデル読み込み
    model = load_model(model_Dir, custom_objects={'dice_coef_loss': dl_loss})

    """評価"""
    # 評価用関数からお手本であるマスク画像を取り出す
    inputs, teachers, fnames = val_generater.generate_4_evaluate()
    # モデルが出力するマスク画像を用意する
    pred_teachers = model.predict(inputs)

    # 比較するマスク画像をセット
    true_masks = []
    pred_masks = []
    for true_mask, pred_mask in zip(teachers, pred_teachers):
        true_masks.append(np.argmax(true_mask, axis=2))
        pred_masks.append(np.argmax(pred_mask, axis=2))

    # MeanIoUの計算
    # calc_OneHotMeanIoU(true_masks=teachers, pred_masks=pred_teachers, num_classes=len(class_ids)+1, data_name='Val')
    calc_MeanIoU(true_masks=true_masks, pred_masks=pred_masks, num_classes=len(class_ids) + 1, data_name="Val")


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    print('\nelapsed time: {:.3f} [sec]'.format(time.perf_counter() - start_time))
    print('Fin.')
