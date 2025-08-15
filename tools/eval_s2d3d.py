from __future__ import print_function

import os
import sys
from PIL import Image
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import numpy as np
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

from segmentron.utils.visualize import get_color_pallete

###############
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import time



#カラーマップの定義（Cityscapesのようなデータセットの場合）
Sfcolor = np.array([
    [255, 0, 0],  # beam
    [255, 128, 0],  # board
    [255, 192, 0],    # bookcase
    [192, 255, 0], # ceiling
    [128, 255, 0], # chair
    [0, 255, 0], # clutter
    [0, 255, 192],  # column
    [0, 192, 255],   # door
    [0, 128, 255],  # floor
    [0, 0, 255], # sofa
    [128, 0, 255],  # table
    [192, 0, 255],   # wall
    [255, 0, 255],     # window
],dtype=float)
Sfcolor[:,:3]/=256 #RGBを0-1の範囲に変換
Sfcmap=ListedColormap(Sfcolor)




NAME_CLASSES = [
    # 'unknown',
    'beam', 'board', 'bookcase', 'ceiling', 'chair',
                'clutter', 'column', 'door', 'floor', 'sofa',
                'table', 'wall', 'window']

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])  
        val_dataset = get_segmentation_dataset('stanford2d3d_pan', split='val', mode='val',
                                               transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=False)
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


    def visualize_segmentation(self,image, output, target, filename, save_dir="results_meiji_indoor"):
        os.makedirs(save_dir, exist_ok=True)

        # 画像をCPUに戻してNumPy配列に変換
        image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        image = image - image.min()  # 負の値を 0 以上にする
        image = image / image.max()  # 最大値を 1 に正規化

        output = output.argmax(0).cpu().numpy()  # モデルの予測結果 (H, W)
        target = target.cpu().numpy()  # ground truth (H, W )

        plt.imshow(output, cmap=Sfcmap, alpha=1.0, vmin=0, vmax=12, interpolation='none')
        plt.axis("off")


        base_filename = os.path.basename(filename)
        save_path = os.path.join(save_dir, f"{base_filename}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()



    def eval(self):
        logging.info("Target eval.")
        # self._eval(self.val_loader)
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        # import time
        # time_start = time.time()
        
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            
            # Start timing
            start_time = time.perf_counter()#計測開始

            with torch.no_grad():
                output = model.evaluate(image)

            # End timing
            end_time = time.perf_counter() #計測終了
            # Calculate and print processing time
            processing_time = (end_time - start_time)*1000
            print(f"Processing time for {filename}: {processing_time:.6f} ms ")
            
            # self.metric.update(output, target)
            # pixAcc, mIoU = self.metric.get()
            # logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #     i + 1, pixAcc * 100, mIoU * 100))

            # # 可視化を実行
            # for j in range(len(filename)):
            #     self.visualize_segmentation(image[i].cpu(), output[i].cpu(), target[i].cpu(), filename[i])

            self.visualize_segmentation(image[0].cpu(), output[0].cpu(), target[0].cpu(), filename[0])
            # self.visualize_segmentation(image[1].cpu(), output[1].cpu(), target[1].cpu(), filename[1])
            # self.visualize_segmentation(image[2].cpu(), output[2].cpu(), target[2].cpu(), filename[2])
            # self.visualize_segmentation(image[3].cpu(), output[3].cpu(), target[3].cpu(), filename[3])







    # def _eval(self, dataloader):
    #     self.metric.reset()
    #     self.model.eval()
    #     if self.args.distributed:
    #         model = self.model.module
    #     else:
    #         model = self.model

    #     logging.info("Start validation, Total sample: {:d}".format(len(dataloader)))
    #     import time
    #     time_start = time.time()
    #     hist = np.zeros((len(NAME_CLASSES), len(NAME_CLASSES)))
    #     for i, (image, target, filename) in enumerate(dataloader):
    #         image = image.to(self.device)
    #         target = target.to(self.device)

    #         with torch.no_grad():
    #             output = model.evaluate(image)

    #         self.metric.update(output, target)
    #         pixAcc, mIoU = self.metric.get()
    #         if i % 10 == 0:
    #             logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
    #                 i + 1, pixAcc * 100, mIoU * 100))

    #     synchronize()
    #     pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
    #     logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
    #     logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
    #             pixAcc * 100, mIoU * 100))

    #     headers = ['class id', 'class name', 'iou']
    #     table = []
    #     for i, cls_name in enumerate(self.classes):
    #         table.append([cls_name, category_iou[i]])
    #     logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
    #                                                        numalign='center', stralign='center')))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
