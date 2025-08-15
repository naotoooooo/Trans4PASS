from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup

######################
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

                
# # カラーマップの定義（Cityscapesのようなデータセットの場合）
# CITYSCAPES_COLORMAP = np.array([
#     [128, 64, 128],  # Road
#     [244, 35, 232],  # Sidewalk
#     [70, 70, 70],    # Building
#     [102, 102, 156], # Wall
#     [190, 153, 153], # Fence
#     [153, 153, 153], # Pole
#     [250, 170, 30],  # Traffic light
#     [220, 220, 0],   # Traffic sign
#     [107, 142, 35],  # Vegetation
#     [152, 251, 152], # Terrain
#     [70, 130, 180],  # Sky
#     [220, 20, 60],   # Person
#     [255, 0, 0],     # Rider
#     [0, 0, 142],     # Car
#     [0, 0, 70],      # Truck
#     [0, 60, 100],    # Bus
#     [0, 80, 100],    # Train
#     [0, 0, 230],     # Motorcycle
#     [119, 11, 32],   # Bicycle
# ])


class Testor(object):
    def __init__(self, args, weather):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])  

        # dataset and dataloader
        # crop_size = cfg.TEST.CROP_SIZE
        # val_dataset = get_segmentation_dataset('synpass', split='val', mode='val', transform=input_transform, weather=weather)
        val_dataset = get_segmentation_dataset('synpass', split='test', mode='val', transform=input_transform, weather=weather)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)
                
                
    def visualize_segmentation(self,image, output, target, filename, save_dir="output_vis"):
        os.makedirs(save_dir, exist_ok=True)

        # 画像をCPUに戻してNumPy配列に変換
        image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        image = image - image.min()  # 負の値を 0 以上にする
        image = image / image.max()  # 最大値を 1 に正規化

        output = output.argmax(0).cpu().numpy()  # モデルの予測結果 (H, W)
        target = target.cpu().numpy()  # ground truth (H, W )
        

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(target, cmap="jet", alpha=0.6)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(output, cmap="jet", alpha=0.6)
        axs[2].set_title("Predicted Segmentation")
        axs[2].axis("off")

        save_path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(save_path)
        plt.close()
        

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model.evaluate(image)

            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
            # 可視化を実行
            self.visualize_segmentation(image[0].cpu(), output[0].cpu(), target[0].cpu(), filename[0])

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))
        return mIoU * 100

class Evaluator(object):
    def __init__(self, args, weather):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])  

        # dataset and dataloader
        # crop_size = cfg.TEST.CROP_SIZE
        val_dataset = get_segmentation_dataset('synpass', split='val', mode='val', transform=input_transform, weather=weather)
        # val_dataset = get_segmentation_dataset('synpass', split='test', mode='val', transform=input_transform, weather=weather)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)



    def visualize_segmentation(self,image, output, target, filename, save_dir="output_vis"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 画像をCPUに戻してNumPy配列に変換
        image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        image = np.clip(image, 0, 255).astype(np.uint8)
        output = output.argmax(0).cpu().numpy()  # モデルの予測結果 (H, W)
        target = target.cpu().numpy()  # ground truth (H, W)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(target, cmap="jet", alpha=0.6)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(output, cmap="jet", alpha=0.6)
        axs[2].set_title("Predicted Segmentation")
        axs[2].axis("off")

        save_path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(save_path)
        plt.close()
        
        
    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model.evaluate(image)

            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))
        return mIoU * 100


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    for weather in ['test']:
        testor = Testor(args, weather)
        m = testor.eval()
        print(weather, m)

    # for weather in ['all', '/cloud', '/fog', '/rain', '/sun']:
    #     evaluator = Evaluator(args, weather)
    #     m = evaluator.eval()
    #     print(weather, m)

    # for weather in ['day', 'night']:
    #     evaluator = Evaluator(args, weather)
    #     m = evaluator.eval()
    #     print(weather, m)
