"""
UAVID dataset test script with optional TTA and RGB mask output
Adapted from loveda_test.py
"""

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from tools.cfg import py2cfg


# UAVID label id → RGB 颜色（论文官方配色）
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]      # Building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]   # Road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]      # Tree
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]    # Low vegetation
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]     # Moving car
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]    # Static car
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]      # Human
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]        # Background
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_png = label2rgb(mask)
        mask_png = cv2.cvtColor(mask_png, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_id + '.png', mask_png)
    else:
        mask_png = mask.astype(np.uint8)
        cv2.imwrite(mask_id + '.png', mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default="config/uavid/DFSAFNet.py")
    arg("-o", "--output_path", type=Path, default="test_results/uavid")
    arg("-t", "--tta", default=None,choices=[None, "d4", "lr"])  # lr: flip TTA, d4: multi-scale TTA
    arg("--rgb", help="Whether output RGB visual masks", action="store_true", default=True)
    arg("--val", help="Whether to evaluate on validation set", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # 加载模型权重
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'),
        config=config,
        strict=False
    )
    model.cuda()
    model.eval()

    # TTA
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    # 数据集选择
    if args.val:
        test_dataset = config.val_dataset
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
    else:
        test_dataset = config.test_dataset  # 无标签的测试集

    results = []

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        for input in tqdm(test_loader):
            # 输出 NxCxHxW
            raw_predictions = model(input['img'].cuda())
            image_ids = input["img_id"]

            if args.val:
                masks_true = input['gt_semantic_seg']

            img_type = input.get('img_type', [''] * raw_predictions.shape[0])

            # Softmax → argmax
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            # 保存每张预测
            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = str(args.output_path / img_type[i] / image_ids[i]) if args.val else str(
                    args.output_path / image_ids[i])
                if args.val:
                    out_dir = os.path.join(args.output_path, img_type[i])
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, mask_name, args.rgb))
                else:
                    results.append((mask, mask_name, args.rgb))

    # 计算验证集指标
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print(f"F1_{class_name}:{class_f1:.4f}, IOU_{class_name}:{class_iou:.4f}")
        print(f"F1:{np.nanmean(f1_per_class):.4f}, mIOU:{np.nanmean(iou_per_class):.4f}, OA:{OA:.4f}")

    # 并行写图
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print(f"Images writing spends: {t1 - t0:.2f} seconds")
    print(f"Model used for testing: {config.test_weights_name}")


if __name__ == "__main__":
    main()
