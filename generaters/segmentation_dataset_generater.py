"""
学習時に教師データから必要な分だけデータを取り出すジェネレーター
・ジェネレーターを使用しないと、教師データを全てメモリに載せる必要があるためマシンスペックを要求される
・元画像はjpg、マスク画像はpngであることを前提としている
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from tensorflow.python.keras.utils import np_utils
from tqdm import tqdm


class SegmentationDataGenerator:
    def __init__(self, input_dir, mask_dir, image_shape, target_class_ids=None, num_classes=None, augment_params={}):
        self.__input_dir = input_dir
        self.__mask_dir = mask_dir
        self.__image_shape = image_shape
        self.__target_class_ids = target_class_ids
        self.__num_classes = num_classes
        if self.__target_class_ids is not None:
            self.__num_classes = len(self.__target_class_ids) + 1
        if self.__num_classes is None:
            self.__print_error('Number of target classes is unknown')
        self.__update_data_names()
        self.file_name = None

    # 存在するデータのみを取得する
    # 公開されているデータの中には、データ欠損が見られる場合がある
    def __update_data_names(self):
        files = glob.glob(os.path.join(self.__mask_dir, '*.png'))
        files.sort()
        self.__data_names = []
        for file in files:
            name = os.path.basename(file)
            if not os.path.exists(self.__input_dir):
                continue
            name, ex = name.split('.')
            self.__data_names.append(name)

    def data_size(self):
        return len(self.__data_names)

    # 学習用ジェネレーター
    # yieldで返すことで学習が終わるまで呼び出される
    def generator(self, batch_size=None):
        if self.__num_classes is None:
            self.__print_error('Number of target classes is unknown')
            return None

        if batch_size is None:
            batch_size = self.data_size()

        input_list = []
        mask_list = []
        while True:
            random.shuffle(self.__data_names)

            for name in self.__data_names:
                self.file_name = name
                input_img, mask_img = self.__load_data(name)
                if input_img is None or mask_img is None:
                    continue

                input_list.append(input_img)
                mask_list.append(mask_img)

                if len(input_list) >= batch_size:
                    inputs = np.array(input_list)
                    masks = np.array(mask_list)
                    input_list = []
                    mask_list = []

                    yield inputs, masks

    # 評価用ジェネレーター(ジェネレーターではない)
    # ファイルからデータを一度だけ読み込み、リストを返す。
    def generate_4_evaluate(self):
        if self.__num_classes is None:
            self.__print_err('Number of target classes is unknown')
            return None, None

        input_list = []
        teacher_list = []
        fname_list = []
        random.shuffle(self.__data_names)

        pbar = tqdm(total=len(self.__data_names), desc="Generate", unit=" data")
        for name in self.__data_names:
            input_img, teacher_img = self.__load_data(name)
            if input_img is None or teacher_img is None:
                continue

            input_list.append(input_img)
            teacher_list.append(teacher_img)
            fname_list.append(os.path.basename(name))
            pbar.update(1)
        pbar.close()

        inputs = np.array(input_list)
        teachers = np.array(teacher_list)

        return inputs, teachers, fname_list

    def __load_data(self, name):
        # 画像読み込み
        input_path = os.path.join(self.__input_dir, name+'.jpg')
        input_img = Image.open(input_path)
        if input_img is None:
            return None, None
        # リサイズして、255で割って正規化
        input_img = input_img.resize(self.__image_shape)
        input_img = np.array(input_img)
        input_img = input_img / 255

        # マスク画像読み込み
        mask_path = os.path.join(self.__mask_dir, name+'.png')
        mask_img = Image.open(mask_path)
        mask_img = mask_img.resize(self.__image_shape)
        mask_img = np.array(mask_img)

        # クラスが指定されていないとき
        # 何の処理か忘れた。たぶんラベルを割り振っている
        if self.__target_class_ids is not None or self.__target_class_ids != []:
            cond = np.logical_not(np.isin(mask_img, self.__target_class_ids))
            mask_img[cond] = 0
            mask_img = mask_img.astype(np.uint16)
            mask_img *= 100
            for k, cls_id in enumerate(self.__target_class_ids):
                mask_img[mask_img == cls_id * 100] = k + 1

        # マスク画像はOne-Hot形式で出力する
        mask_img = np_utils.to_categorical(mask_img, num_classes=self.__num_classes)

        return input_img, mask_img

    def __print_error(self, error_str):
        print('<SegmentationDataGenerator> Error :', error_str)


# 正しく読み込めるかの確認
if __name__ == '__main__':
    target_base_Dir = os.path.join('.', 'datasets/VOC2012/target_data')
    class_base_Dir = os.path.join(target_base_Dir, 'person_')
    train_imgs_Dir = os.path.join(class_base_Dir, 'train', 'img')
    train_segs_Dir = os.path.join(class_base_Dir, 'train', 'seg')
    train_generater = SegmentationDataGenerator(input_dir=train_imgs_Dir, mask_dir=train_segs_Dir,
                                                image_shape=(125, 125), target_class_ids=[2, 7, 9, 15])
    print(train_generater.data_size())
    train_dataset = train_generater.generator(batch_size=16)
    for i, data in enumerate(train_dataset):
        if i > 2:
            break
        for d in data:
            print(d[0].shape)

