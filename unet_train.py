"""
UNetモデルの学習
・教師データはあらかじめ準備したVOC 2012であることを前提としている。
・別の教師データを使用する場合は、ファイル指定を調整してください。
"""
from kerasmodels import UNet, UNet_with_batchnorm
import os
from generaters import SegmentationDataGenerator
import math
import time
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tools import DiceLoss
from keras.optimizers.optimizer_v2.adam import Adam


def main():
    """データ準備"""
    # VOC 2012から学習するラベルを選択
    IDs = {'person': 15, 'bike': 14, 'car': 7, 'bicycle': 2}
    class_ids = [IDs['person'], IDs['car']]
    # class_ids = [IDs['person']]

    # あらかじめ用意したVOC 2012のpersonラベルを含む画像ファイル群のtrainデータの読み込み
    target_base_Dir = os.path.join('.', 'datasets/VOC2012/target_data')
    class_base_Dir = os.path.join(target_base_Dir, 'person_')
    train_imgs_Dir = os.path.join(class_base_Dir, 'train', 'img')
    train_segs_Dir = os.path.join(class_base_Dir, 'train', 'seg')
    # trainデータをジェネレーターにセットする
    train_generator = SegmentationDataGenerator(input_dir=train_imgs_Dir, mask_dir=train_segs_Dir,
                                                image_shape=(128, 128), target_class_ids=class_ids)

    # validationデータの読み込み
    val_imgs_Dir = os.path.join(class_base_Dir, 'validation', 'img')
    val_segs_Dir = os.path.join(class_base_Dir, 'validation', 'seg')
    # ジェネレーターにセット
    val_generater = SegmentationDataGenerator(input_dir=val_imgs_Dir, mask_dir=val_segs_Dir,
                                              image_shape=(128, 128), target_class_ids=class_ids)

    """モデル準備"""
    MODELNAME = 'unetBNDL_PC'
    BATCH_SIZE = 16
    EPOCH = 300
    input_shape = (128, 128, 3)
    class_num = len(class_ids) + 1
    # モデル読み込み
    model = UNet_with_batchnorm(input_shape=input_shape, class_num=class_num)

    # 最適化アルゴリズムの設定
    optimizer = Adam(lr=0.01)
    # dl_loss = 'categorical_crossentropy'
    dl_loss = DiceLoss(input_shape=input_shape, class_num=class_num).dice_coef_loss

    # コンパイル
    model.compile(optimizer=optimizer,
                  loss=dl_loss,
                  metrics=['acc'])

    # TensorBoardによるログの設定
    log_Dir = os.path.join('.', 'Logs', MODELNAME)
    if not os.path.exists(log_Dir):
        os.makedirs(log_Dir)
    tensorboard_callback = TensorBoard(log_dir=log_Dir)

    # 指定したエポックごとに評価値を確認し、精度が上がっていればその時点のモデルを保存する
    save_Dir = os.path.join('./results', MODELNAME, 'learned_model_best')
    if not os.path.exists(save_Dir):
        os.makedirs(save_Dir)
    model_save_callback = ModelCheckpoint(filepath=save_Dir, verbose=1, save_best_only=True, period=5)
    # 評価値を監視し、改善が見られなかったら学習率を下げる
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1)

    """学習"""
    # ジェネレータを使用する場合は,step_per_epochを設定する
    model.fit(
        x=train_generator.generator(batch_size=BATCH_SIZE),
        steps_per_epoch=math.ceil(train_generator.data_size() / BATCH_SIZE),
        batch_size=BATCH_SIZE,
        validation_data=val_generater.generator(batch_size=BATCH_SIZE),
        validation_steps=math.ceil(val_generater.data_size() / BATCH_SIZE),
        epochs=EPOCH,
        callbacks=[tensorboard_callback, lr_scheduler, model_save_callback],
        verbose=1
    )

    """学習後"""
    save_Dir = os.path.join('./results', MODELNAME, 'learned_model')
    if not os.path.exists(save_Dir):
        os.makedirs(save_Dir)
    model.save(save_Dir)


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.perf_counter() - start_time))
    print('\nFin.')
