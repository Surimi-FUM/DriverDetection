"""
推論時に指定したファイルから必要な分だけデータを取り出すジェネレーター
・ジェネレーターを使用しないと、教師データを全てメモリに載せる必要があるためマシンスペックを要求される
・元画像はjpg、マスク画像はpngであることを前提としている
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import random


class PredictDataGenerator:
    def __init__(self, input_dir, image_shape):
        self.__input_dir = input_dir
        self.__image_shape = image_shape
        self.__get_img_names()

    # 存在するデータのみを取得する
    # 公開されているデータの中には、データ欠損が見られる場合がある
    def __get_img_names(self):
        files = glob.glob(os.path.join(self.__input_dir, '*'))
        files.sort()
        self.__data_names = []
        for file in files:
            name = os.path.basename(file)
            if not os.path.exists(self.__input_dir):
                continue
            name, ex = name.split('.')
            self.__data_names.append(name)

    # ジェネレーター
    # テスト画像の場合、正解マスクがないため1つだけ返す
    def generator(self, shuffle=False):
        if shuffle:
            random.shuffle(self.__data_names)

        for name in self.__data_names:
            test_img = self.__load_data(name)
            if test_img is None:
                continue
            yield np.array([test_img])

    def __load_data(self, name):
        input_path = os.path.join(self.__input_dir, name+'.jpg')
        input_img = Image.open(input_path)
        if input_img is None:
            return None
        # リサイズして、255で割って正規化
        input_img = input_img.resize(self.__image_shape)
        input_img = np.array(input_img)
        input_img = input_img / 255

        return input_img


# 正しく読み込めるかの確認
if __name__ == '__main__':
    test_imgs_Dir = '../Dataset/test_img'
    train_generater = PredictDataGenerator(input_dir=test_imgs_Dir, image_shape=(128, 128))
    test_dataset = train_generater.generator(shuffle=False)

    for i, data in enumerate(test_dataset):

        print(i, data.shape)
        plt.imshow(data[0])
        plt.show()
