"""
学習済みモデルによる推論
・モデルの
"""
import time
import os
from keras.models import load_model
from generaters import PredictDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tools import ColorMap
from tools import DiceLoss
import cv2


def main():
    """データ準備"""
    # モデルの出力層のフィルタ数を求めるのに使う
    # 直で指定してもよいが、学習時と同じ書式にすることでミスを減らせると思う
    IDs = {'person': 15, 'bike': 14, 'car': 7, 'bicycle': 2}
    # class_ids = [IDs['person'], IDs['car'], IDs['bike']]
    class_ids = [IDs['person']]

    # テストデータの読み込み
    test_imgs_Dir = './datasets/test_img'
    test_generater = PredictDataGenerator(input_dir=test_imgs_Dir, image_shape=(128, 128))

    # 教師データを推論することで、どのように学習ができているのかを確認できる
    target_base_Dir = os.path.join('.', 'datasets/VOC2012/target_data')
    class_base_Dir = os.path.join(target_base_Dir, 'person_')
    train_imgs_Dir = os.path.join(class_base_Dir, 'train', 'img')
    train_generater = PredictDataGenerator(input_dir=train_imgs_Dir, image_shape=(128, 128))

    """モデル準備"""
    input_shape = (128, 128, 3)
    class_num = len(class_ids) + 1
    MODELNAME = 'unetBNDL_P'
    model_Dir = os.path.join('.', 'results', MODELNAME, 'learned_model_best')
    dl_loss = DiceLoss(input_shape=input_shape, class_num=class_num).dice_coef_loss
    # モデル読み込み。DiceLossは自作関数なため、custom_objectsに指定しなければ読み込めない
    model = load_model(model_Dir, custom_objects={'dice_coef_loss': dl_loss})

    """予測・結果表示"""
    color_map = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]
    for i, test_img in enumerate(test_generater.generator(shuffle=True)):
        if i > 2:
            break
        # 推論
        predict = model.predict(test_img)
        # model.predictの結果はOne-Hotベクトル。これをargmaxでクラスIDに変換する
        pred_argmax = np.argmax(predict[0], axis=2)
        # カラーマップにしたがってクラスIDをカラー画像に変換
        h, w = np.shape(predict[0])[:2]
        if not bool(color_map):
            color_map = ColorMap(n=len(class_ids)).get_list()
        pred_color = np.zeros((h, w, 3), dtype=np.uint8)
        for j, _ in enumerate(class_ids):
            pred_color[pred_argmax == (j + 1)] = color_map[j % len(color_map)]

        # 推論結果を元画像に重ねる
        org_img = (test_img[0] * 255).astype(np.uint8)
        pred_on_img = cv2.addWeighted(org_img, 1, pred_color, 0.6, 0)

        # 表示
        fig, ax = plt.subplots(1, 3)
        ax[0].axis('off')
        ax[0].set_title('Original')
        ax[0].imshow(test_img[0])
        ax[1].axis('off')
        ax[1].set_title('Predict')
        ax[1].imshow(pred_color)
        ax[2].axis('off')
        ax[2].set_title('Overlap')
        ax[2].imshow(pred_on_img)
        plt.show()

        # 推論結果の保存
        data_type = 'learned'
        save_dir = os.path.join(model_Dir, data_type+'_results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # fig.savefig(os.path.join(save_dir, data_type+f'{i+1}.jpg'))


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.perf_counter() - start_time))
    print('\nFin.')
