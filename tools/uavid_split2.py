import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
import random

# 固定随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ======================= UAVID 调色板 =======================
Building = np.array([128, 0, 0])       # label 0
Road = np.array([128, 64, 128])        # label 1
Tree = np.array([0, 128, 0])           # label 2
LowVeg = np.array([128, 128, 0])       # label 3
Moving_Car = np.array([64, 0, 128])    # label 4
Static_Car = np.array([192, 0, 192])   # label 5
Human = np.array([64, 64, 0])          # label 6
Clutter = np.array([0, 0, 0])          # label 7
Boundary = np.array([255, 255, 255])   # label 255

num_classes = 8


# 参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="D:\\pbyy\\GeoSeg-main\\data\\uavid\\val_val")
    parser.add_argument("--output-img-dir", default="C:\\Users\\zwd\\Desktop\\val\\images")
    parser.add_argument("--output-mask-dir", default="C:\\Users\\zwd\\Desktop\\val\\masks")
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--split-size-h", type=int, default=1024)
    parser.add_argument("--split-size-w", type=int, default=1024)
    parser.add_argument("--stride-h", type=int, default=1024)
    parser.add_argument("--stride-w", type=int, default=1024)
    return parser.parse_args()


# label id → RGB
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = Tree
    mask_rgb[np.all(mask_convert == 3, axis=0)] = LowVeg
    mask_rgb[np.all(mask_convert == 4, axis=0)] = Moving_Car
    mask_rgb[np.all(mask_convert == 5, axis=0)] = Static_Car
    mask_rgb[np.all(mask_convert == 6, axis=0)] = Human
    mask_rgb[np.all(mask_convert == 7, axis=0)] = Clutter
    mask_rgb[np.all(mask_convert == 255, axis=0)] = Boundary
    return mask_rgb


# RGB → label id
def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0
    label_seg[np.all(label == Road, axis=-1)] = 1
    label_seg[np.all(label == Tree, axis=-1)] = 2
    label_seg[np.all(label == LowVeg, axis=-1)] = 3
    label_seg[np.all(label == Moving_Car, axis=-1)] = 4
    label_seg[np.all(label == Static_Car, axis=-1)] = 5
    label_seg[np.all(label == Human, axis=-1)] = 6
    label_seg[np.all(label == Clutter, axis=-1)] = 7
    label_seg[np.all(label == Boundary, axis=-1)] = 255
    return label_seg


# 数据增强封装
def image_augment(image, mask, mode='train'):
    image_list = []
    mask_list = []
    if mode == 'train':
        mask_tmp = rgb2label(mask)
        image_list.append(image)
        mask_list.append(mask_tmp)
    else:
        mask_tmp = rgb2label(mask)
        image_list.append(image)
        mask_list.append(mask_tmp)
    return image_list, mask_list, [mask]  # 返回原彩色 mask 方便保存


# pad 填充
def padifneeded(image, mask):
    pad = albu.PadIfNeeded(min_height=2160, min_width=4096, position='bottom_right',
                           border_mode=0, value=[0, 0, 0], mask_value=[255, 255, 255])(image=image, mask=mask)
    img_pad, mask_pad = pad['image'], pad['mask']
    return img_pad, mask_pad


# 切 patch
def patch_format(inp):
    (input_dir, seq, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp
    img_paths = glob.glob(os.path.join(input_dir, str(seq), 'Images', "*.png"))
    mask_paths = glob.glob(os.path.join(input_dir, str(seq), 'Labels', "*.png"))

    gray_dir = os.path.join(masks_output_dir, "gray")
    rgb_dir = os.path.join(masks_output_dir, "rgb")
    os.makedirs(gray_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_rgb_full = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_rgb_full = cv2.cvtColor(mask_rgb_full, cv2.COLOR_BGR2RGB)
        id = os.path.splitext(os.path.basename(img_path))[0]

        img, mask_rgb_full = padifneeded(img, mask_rgb_full)
        image_list, mask_id_list, mask_rgb_list = image_augment(img, mask_rgb_full, mode=mode)

        for m in range(len(image_list)):
            k = 0
            img = image_list[m]
            mask_id = mask_id_list[m]
            mask_rgb = mask_rgb_list[0]  # 保证彩色对照

            for y in range(0, img.shape[0], stride[0]):
                for x in range(0, img.shape[1], stride[1]):
                    img_tile = img[y:y+split_size[0], x:x+split_size[1]]
                    mask_tile_gray = mask_id[y:y+split_size[0], x:x+split_size[1]]
                    mask_tile_rgb = mask_rgb[y:y+split_size[0], x:x+split_size[1]]

                    if img_tile.shape[:2] == split_size:
                        # 保存图像
                        out_img_path = os.path.join(imgs_output_dir, f"{seq}_{id}_{m}_{k}.png")
                        cv2.imwrite(out_img_path, cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))

                        # 保存灰度 ID 标签
                        out_mask_path = os.path.join(gray_dir, f"{seq}_{id}_{m}_{k}.png")
                        cv2.imwrite(out_mask_path, mask_tile_gray)

                        # 保存彩色标签
                        out_mask_rgb_path = os.path.join(rgb_dir, f"{seq}_{id}_{m}_{k}.png")
                        cv2.imwrite(out_mask_rgb_path, cv2.cvtColor(mask_tile_rgb, cv2.COLOR_RGB2BGR))

                    k += 1


# 主程序
if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    split_size = (args.split_size_h, args.split_size_w)
    stride = (args.stride_h, args.stride_w)
    seqs = os.listdir(args.input_dir)

    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    inp = [(args.input_dir, seq, args.output_img_dir, args.output_mask_dir,
            args.mode, split_size, stride) for seq in seqs]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    print(f'Images splitting spends: {t1:.2f} seconds')
