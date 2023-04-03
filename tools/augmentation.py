"""
データオーグメンテーション（データ拡張）
・教師データに画像処理を施して、データ数を水増しする
・ここでは画像の拡大縮小を実施する
・別ファイルから呼び出して使用するプログラムではない
"""
import os
import glob
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import time


def augment(input_imgs_dir, input_masks_dir, out_imgs_dir, out_masks_dir, color_erase_num=2, gray_erase_num=2):
    files = glob.glob(os.path.join(input_masks_dir, '*.png'))
    files.sort()

    pbar = tqdm(total=len(files), desc='Augment', unit=' Files')
    for file in files:
        name, _ = os.path.basename(file).split('.')
        input_path = os.path.join(input_imgs_dir, name+'.jpg')

        if not os.path.exists(input_path):
            print('skip : ', input_path)
            pbar.update(1)
            continue

        # データ拡張
        input_img = cv2.imread(input_path)
        mask_img = np.array(Image.open(file))
        for _ in range(1, color_erase_num):
            imgs, e_infos = random_erase(input_img, mask_img)
            save_erase_imgs(name, out_imgs_dir, out_masks_dir, imgs, e_infos)

        # グレイスケールにして、さらにデータ拡張
        gray_input_img = convert_colorbgr2graybgr(input_img)
        gray_mask_img = mask_img
        for _ in range(1, gray_erase_num):
            imgs, e_infos = random_erase(gray_input_img, gray_mask_img)
            save_erase_imgs(name, out_imgs_dir, out_masks_dir, imgs, e_infos)

        pbar.update(1)
    pbar.close()


# データ拡張処理
def random_erase(input_base_img, mask_base_img, erase_pos_min=10, erase_pos_max=400,
                 erase_size_min=120, erase_size_max=120):
    # 画像を拡大縮小
    epos_h = random.randint(erase_size_min, erase_size_max)
    epos_w = random.randint(erase_size_min, erase_size_max)
    esize_h = random.randint(erase_size_min, erase_size_max)
    esize_w = random.randint(erase_size_min, erase_size_max)

    erase_mask = mask_base_img.copy()
    erase_input = input_base_img.copy()
    # 0パティング
    erase_mask[epos_h:epos_h+esize_h, epos_w:epos_w+esize_w] = 0
    erase_input[epos_h:epos_h + esize_h, epos_w:epos_w + esize_w, :] = 0

    return (erase_input, erase_mask), ((epos_h, epos_w), (esize_h, esize_w))


def save_erase_imgs(name, out_img_dir, out_mask_dir, imgs, e_infos):
    erase_img, erase_mask = imgs
    (epos_h, epos_w), _ = e_infos

    aug_fname = f'{name}_h{epos_h}_w{epos_w}'

    out_mask_path = os.path.join(out_mask_dir, aug_fname+'.png')
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)

    out_img_path = os.path.join(out_img_dir, aug_fname+'.jpg')
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    Image.fromarray(erase_mask).save(out_mask_path)
    cv2.imwrite(out_img_path, erase_img)


def convert_colorbgr2graybgr(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    graybgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return graybgr_img


def main():
    def SRC_DIR(base, data_type):
        return os.path.join(base, 'validation', data_type)

    def SAVE_DIR(base, data_type):
        return os.path.join(base, 'augmented_validation', data_type)

    # データ拡張するデータを指定
    BASE_DATA_DIR = os.path.join('.', 'datasets/VOC2012/target_data')
    all_files = glob.glob(os.path.join(BASE_DATA_DIR, '*'))
    data_type = ('img', 'seg')
    # データ拡張
    for file_dir in all_files:
        augment(SRC_DIR(file_dir, data_type[0]), SRC_DIR(file_dir, data_type[1]),
                SAVE_DIR(file_dir, data_type[0]), SAVE_DIR(file_dir, data_type[1]))


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.perf_counter() - start_time))
    print('\nFin.')
