"""
VOC2012を教師データとして整理するプログラム
・ここでは以下の処理が行える
1.VOC2012から特定のクラスIDを含む画像のみを別ファイルに分ける
2.教師データを指定した比率の枚数で画像を分ける
・別ファイルから呼び出されることはない
"""
import glob
import os
import shutil
import time
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

# 1. クラスID毎にシャッフルして分類
def classify_ID():
    save_DIR = os.path.join('.', 'datasets', 'VOC2012', 'target_data')
    IDs = {'person': 15, 'bike': 14, 'car': 7, 'bicycle': 2}
    SEG_DIR = 'D:/VOC2012/SegmentationClass'
    IMG_DIR = 'D:/VOC2012/JPEGImages'

    segs = glob.glob(os.path.join(SEG_DIR, '*.png'))
    segs.sort()

    cls_counters = {}
    for seg_file in segs:
        fname, _ = os.path.splitext(os.path.basename(seg_file))
        img_file = os.path.join(IMG_DIR, fname + '.jpg')
        if not os.path.exists(img_file):
            print('not found :', img_file)

        seg_img = np.array(Image.open(seg_file))

        copy_to = ''
        # 特定のIDがある画像ならコピー処理を行う
        if np.any(seg_img == IDs['person']):
            copy_to += 'person_'

        # 特定のIDがないならその画像をスルーする
        if copy_to == '':
            continue

        # ファイル名を連番で付ける
        if not copy_to in cls_counters.keys():
            cls_counters[copy_to] = 0
        cls_counters[copy_to] += 1

        # マスク画像と元画像を保存
        copy_seg_path = os.path.join(save_DIR, copy_to, 'seg')
        if not os.path.exists(copy_seg_path):
            os.makedirs(copy_seg_path)
        shutil.copy(seg_file, copy_seg_path)

        copy_img_path = os.path.join(save_DIR, copy_to, 'img')
        if not os.path.exists(copy_img_path):
            os.makedirs(copy_img_path)
        shutil.copy(img_file, copy_img_path)

    print(cls_counters)


# 2. 画像枚数を比率に従って分ける
def separate_train_validation():
    data_base_Dir = os.path.join('.', 'datasets', 'VOC2012', 'target_data', 'all_')

    # trainファイル
    train_base_Dir = os.path.join(data_base_Dir, 'train')
    train_seg_Dir = os.path.join(train_base_Dir, 'seg')
    train_img_Dir = os.path.join(train_base_Dir, 'img')

    # validationファイル
    validation_base_Dir = os.path.join(data_base_Dir, 'validation')
    validation_seg_Dir = os.path.join(validation_base_Dir, 'seg')
    validation_img_Dir = os.path.join(validation_base_Dir, 'img')

    # 画像読み込み
    segs = glob.glob(os.path.join(data_base_Dir, 'seg', '*.png'))
    segs.sort()

    # 画像を振り分ける。8:2ならtest_size=0.2
    train_segs, val_segs = train_test_split(segs, test_size=0.2)

    for seg_file in train_segs:
        fname, _ = os.path.splitext(os.path.basename(seg_file))
        img_file = os.path.join(data_base_Dir, 'img', fname + '.jpg')
        if not os.path.exists(img_file):
            print('<train> not found :', img_file)

        if not os.path.exists(train_seg_Dir):
            os.makedirs(train_seg_Dir)
        shutil.copy(seg_file, train_seg_Dir)

        if not os.path.exists(train_img_Dir):
            os.makedirs(train_img_Dir)
        shutil.copy(img_file, train_img_Dir)

    for seg_file in val_segs:
        fname, _ = os.path.splitext(os.path.basename(seg_file))
        img_file = os.path.join(data_base_Dir, 'img', fname + '.jpg')
        if not os.path.exists(img_file):
            print('<val> not found :', img_file)

        if not os.path.exists(validation_seg_Dir):
            os.makedirs(validation_seg_Dir)
        shutil.copy(seg_file, validation_seg_Dir)

        if not os.path.exists(validation_img_Dir):
            os.makedirs(validation_img_Dir)
        shutil.copy(img_file, validation_img_Dir)


if __name__ == '__main__':
    start_time = time.perf_counter()
    separate_train_validation()
    print('elapsed time: {:.3f} [sec]'.format(time.perf_counter() - start_time))
    print('\nFin.')
