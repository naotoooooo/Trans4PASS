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

###############
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import time
from live_ros2_dataset import LiveROS2SegmentationDataset


#カラーマップの定義（Cityscapesのようなデータセットの場合）
Cscolor = np.array([
    [128, 64, 128],  # Road
    [244, 35, 232],  # Sidewalk
    [70, 70, 70],    # Building
    [102, 102, 156], # Wall
    [190, 153, 153], # Fence
    [153, 153, 153], # Pole
    [250, 170, 30],  # Traffic light
    [220, 220, 0],   # Traffic sign
    [107, 142, 35],  # Vegetation
    [152, 251, 152], # Terrain
    [70, 130, 180],  # Sky
    [220, 20, 60],   # Person
    [255, 0, 0],     # Rider
    [0, 0, 142],     # Car
    [0, 0, 70],      # Truck
    [0, 60, 100],    # Bus
    [0, 80, 100],    # Train
    [0, 0, 230],     # Motorcycle
    [119, 11, 32],   # Bicycle
],dtype=float)
Cscolor[:,:3]/=256 #RGBを0-1の範囲に変換
Cscmap=ListedColormap(Cscolor)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])  

        # dataset and dataloader
        # crop_size = cfg.TEST.CROP_SIZE
        # val_dataset = get_segmentation_dataset('densepass', split='val', mode='val', transform=input_transform)
        # val_dataset = get_segmentation_dataset('stanford2d3d_pan', split='trainval', mode='val', transform=input_transform)
        
        
        # topicを指定（ros）
        val_dataset = LiveROS2SegmentationDataset(
            img_topic='/color/image_raw',
            mask_topic=None,                # 学習ならマスクトピックを指定
            crop_size=(331, 1280),
            transform=input_transform
        )
        
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1, drop_last=False)
        
        
        # self.val_loader = data.DataLoader(dataset=val_dataset,
        #                                   batch_sampler=val_batch_sampler,
        #                                   num_workers=cfg.DATASET.WORKERS,
        #                                   pin_memory=True)
        
        # Dataloader は num_workers=0 で固定（ros）
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,              # ライブでは1枚ずつ
            num_workers=0,              # rclpy は fork 非対応
            pin_memory=True
        )

        
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

    def visualize_segmentation(self,image, output, target, filename, save_dir="aaaaaaaaaaaa"):
        os.makedirs(save_dir, exist_ok=True)

        # 画像をCPUに戻してNumPy配列に変換
        image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        image = image - image.min()  # 負の値を 0 以上にする
        image = image / image.max()  # 最大値を 1 に正規化

        output = output.argmax(0).cpu().numpy()  # モデルの予測結果 (H, W)
        target = target.cpu().numpy()  # ground truth (H, W )
        
        # output = output - output.min()  # 負の値を 0 以上にする
        # output = output / output.max()  # 最大値を 1 に正規化
        # target = target - target.min()  # 負の値を 0 以上にする
        # target = target / target.max()  # 最大値を 1 に正規化


        plt.imshow(output, cmap=Cscmap, alpha=1.0, vmin=0, vmax=18, interpolation='none')
        plt.axis("off")

        save_path = os.path.join(save_dir, f"{filename}_seg_only.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # fig, axs = plt.subplots(2, 1, figsize=(16, 4))

        # axs[0].imshow(image)
        # axs[0].set_title("Input Image")
        # axs[0].axis("off")

        # axs[1].imshow(output, cmap=Cscmap, alpha=1.0,vmin=0,vmax=18,interpolation='none')
        # axs[1].set_title("Predicted Segmentation")
        # axs[1].axis("off")
        

        # save_path = os.path.join(save_dir, f"{filename}.png")
        # plt.savefig(save_path)
        # plt.close()
        
        
    def eval(self):
        # self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        # logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        # import time
        # time_start = time.time()
        # IterableDataset は件数不明
        logging.info("Start validation (streaming input; length unknown)")
        
        
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            if target.numel() > 0:   # マスクが空でなければ
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
        # synchronize()
        # pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        # logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        # logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
        #         pixAcc * 100, mIoU * 100))

        # headers = ['class id', 'class name', 'iou']
        # table = []
        # for i, cls_name in enumerate(self.classes):
        #     table.append([cls_name, category_iou[i]])
        # logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
        #                                                    numalign='center', stralign='center')))


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


###########################################################################################################

# # tools/eval_dp.py
# from __future__ import print_function

# import os
# import sys
# import time
# import logging

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import torch.nn.functional as F

# from tabulate import tabulate
# from torchvision import transforms

# # Segmentron / Trans4PASS
# from segmentron.models.model_zoo import get_segmentation_model
# from segmentron.utils.score import SegmentationMetric
# from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
# from segmentron.config import cfg
# from segmentron.utils.options import parse_args
# from segmentron.utils.default_setup import default_setup

# # === ROS2 publish 用 ===
# import rclpy
# from rclpy.node import Node
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image as RosImage

# # 可視化
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# # ライブ入力 Dataset
# from live_ros2_dataset import LiveROS2SegmentationDataset


# # ------------------------------------------------------------
# # カラーマップ定義（Cityscapes trainId=0..18）
# # ------------------------------------------------------------
# Cscolor = np.array([
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
# ], dtype=float)
# Cscolor[:, :3] /= 256.0
# Cscmap = ListedColormap(Cscolor)


# def colorize_trainid(train_id_map: np.ndarray) -> np.ndarray:
#     """(H,W) uint8 の trainId マップを (H,W,3) uint8 (RGB) に変換"""
#     palette = np.array([
#         [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
#         [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
#         [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
#         [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32]
#     ], dtype=np.uint8)
#     h, w = train_id_map.shape
#     color = np.zeros((h, w, 3), dtype=np.uint8)
#     valid = (train_id_map < len(palette))
#     color[valid] = palette[train_id_map[valid]]
#     return color  # RGB


# class SegPublisher(Node):
#     """推論結果（trainId とカラー）を配信する ROS2 ノード"""
#     def __init__(self, label_topic='/segmentation/label', color_topic='/segmentation/color'):
#         super().__init__('segmentation_publisher')
#         self.bridge = CvBridge()
#         self.pub_label = self.create_publisher(RosImage, label_topic, 10)
#         self.pub_color = self.create_publisher(RosImage, color_topic, 10)

#     def publish_label_and_color(self, label_np: np.ndarray, color_rgb_np: np.ndarray,
#                                 stamp_ns: int = None, frame_id: str = 'camera'):
#         # label: mono8, color: bgr8 で RViz2 表示互換
#         msg_label = self.bridge.cv2_to_imgmsg(label_np.astype(np.uint8), encoding='mono8')
#         msg_color = self.bridge.cv2_to_imgmsg(color_rgb_np[:, :, ::-1].copy(), encoding='bgr8')  # RGB→BGR

#         if stamp_ns is not None:
#             sec = int(stamp_ns // 1_000_000_000)
#             nsec = int(stamp_ns % 1_000_000_000)
#             msg_label.header.stamp.sec = sec; msg_label.header.stamp.nanosec = nsec
#             msg_color.header.stamp.sec = sec; msg_color.header.stamp.nanosec = nsec
#         else:
#             now = self.get_clock().now().to_msg()
#             msg_label.header.stamp = now
#             msg_color.header.stamp = now

#         msg_label.header.frame_id = frame_id
#         msg_color.header.frame_id = frame_id
#         self.pub_label.publish(msg_label)
#         self.pub_color.publish(msg_color)


# class Evaluator(object):
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device(args.device)

#         # image transform
#         input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
#         ])

#         # ===== Live ROS2 Dataset & DataLoader =====
#         self.val_dataset = LiveROS2SegmentationDataset(
#             img_topic='/color/image_raw',   # ★ここを実トピックに合わせて変更
#             mask_topic=None,                # 学習/評価でGTがtopicにあればそのトピック名を指定
#             crop_size=(331, 1280),
#             transform=input_transform
#         )
#         # IterableDataset なので batch_sampler は使わない。ROS2 のため num_workers=0 固定
#         self.val_loader = data.DataLoader(
#             dataset=self.val_dataset,
#             batch_size=1,
#             num_workers=0,
#             pin_memory=True
#         )

#         self.classes = self.val_dataset.classes

#         # ===== Network =====
#         self.model = get_segmentation_model().to(self.device)
#         if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and cfg.MODEL.BN_EPS_FOR_ENCODER:
#             logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
#             self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

#         if args.distributed:
#             self.model = nn.parallel.DistributedDataParallel(
#                 self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
#             )
#         self.model.to(self.device)

#         self.metric = SegmentationMetric(self.val_dataset.num_class, args.distributed)

#     def set_batch_norm_attr(self, named_modules, attr, value):
#         for m in named_modules:
#             if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
#                 setattr(m[1], attr, value)

#     def visualize_segmentation(self, image, output, target, filename, save_dir="seg_results"):
#         os.makedirs(save_dir, exist_ok=True)

#         # モデル出力→trainId
#         output = output.argmax(0).cpu().numpy()  # (H, W), 0..18

#         # 画像は 0-1 正規化
#         image = image.cpu().numpy().transpose(1, 2, 0)
#         image = image - image.min()
#         image = image / max(image.max(), 1e-6)

#         # 可視化（予測のみ）
#         plt.imshow(output, cmap=Cscmap, alpha=1.0, vmin=0, vmax=18, interpolation='none')
#         plt.axis("off")
#         save_path = os.path.join(save_dir, f"{filename}_seg_only.png")
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         plt.close()

#     def eval(self):
#         # self.metric.reset()
#         self.model.eval()
#         model = self.model.module if self.args.distributed else self.model

#         logging.info("Start validation (streaming input; length unknown)")

#         # ---- ROS2 publisher 初期化 ----
#         if not rclpy.ok():
#             rclpy.init(args=None)
#         seg_pub = SegPublisher(label_topic='/segmentation/label',
#                                color_topic='/segmentation/color')

#         try:
#             for i, (image, target, filename) in enumerate(self.val_loader):
#                 # image: [B,C,H,W], target: [] or [B,H,W], filename: list[str] (batch_size=1)
#                 image = image.to(self.device)
#                 has_target = (isinstance(target, torch.Tensor) and target.numel() > 0)
#                 if has_target:
#                     target = target.to(self.device)

#                 # 推論
#                 t0 = time.perf_counter()
#                 with torch.no_grad():
#                     output = model.evaluate(image)  # [B,C,H,W]
#                 dt_ms = (time.perf_counter() - t0) * 1000.0
#                 print(f"Processing time for {filename}: {dt_ms:.3f} ms")

#                 # バッチ1前提
#                 out_b = output[0].cpu()
#                 pred = out_b.argmax(0).numpy().astype(np.uint8)  # (H,W) trainId
#                 color = colorize_trainid(pred)                    # (H,W,3) RGB

#                 name0 = filename[0] if isinstance(filename, (list, tuple)) else filename
#                 tgt0 = (target[0].cpu() if has_target
#                         else torch.zeros_like(torch.from_numpy(pred), dtype=torch.long))
#                 self.visualize_segmentation(image[0].cpu(), out_b, tgt0, name0)

#                 # stamp 継承（filename が "<stamp>.png" の場合）
#                 stamp_ns = None
#                 try:
#                     base = str(name0)
#                     if base.endswith('.png'):
#                         base = base[:-4]
#                     if base.isdigit():
#                         stamp_ns = int(base)
#                 except Exception:
#                     stamp_ns = None

#                 # 配信
#                 seg_pub.publish_label_and_color(pred, color, stamp_ns=stamp_ns, frame_id='camera')

#         finally:
#             # クリーンアップ
#             try:
#                 seg_pub.destroy_node()
#             finally:
#                 if rclpy.ok():
#                     rclpy.shutdown()
#             if hasattr(self.val_loader.dataset, "close"):
#                 self.val_loader.dataset.close()


# if __name__ == '__main__':
#     # パス設定
#     cur_path = os.path.abspath(os.path.dirname(__file__))
#     root_path = os.path.split(cur_path)[0]
#     sys.path.append(root_path)

#     args = parse_args()
#     cfg.update_from_file(args.config_file)
#     cfg.update_from_list(args.opts)
#     cfg.PHASE = 'test'
#     cfg.ROOT_PATH = root_path
#     cfg.check_and_freeze()

#     default_setup(args)

#     evaluator = Evaluator(args)
#     evaluator.eval()



###########################################################################################################



# # tools/eval_dp.py
# from __future__ import print_function

# import os
# import sys
# import time
# import logging

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import torch.nn.functional as F

# from tabulate import tabulate
# from torchvision import transforms

# # Segmentron / Trans4PASS
# from segmentron.models.model_zoo import get_segmentation_model
# from segmentron.utils.score import SegmentationMetric
# from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
# from segmentron.config import cfg
# from segmentron.utils.options import parse_args
# from segmentron.utils.default_setup import default_setup

# # === ROS2 publish 用（カラーのみ配信） ===
# import rclpy
# from rclpy.node import Node
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image as RosImage

# # 可視化
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# # ライブ入力 Dataset
# from live_ros2_dataset import LiveROS2SegmentationDataset


# # ------------------------------------------------------------
# # カラーマップ定義（Cityscapes trainId=0..18）
# # ------------------------------------------------------------
# Cscolor = np.array([
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
# ], dtype=float)
# Cscolor[:, :3] /= 256.0
# Cscmap = ListedColormap(Cscolor)


# def colorize_trainid(train_id_map: np.ndarray) -> np.ndarray:
#     """(H,W) uint8 の trainId マップを (H,W,3) uint8 (RGB) に変換"""
#     palette = np.array([
#         [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
#         [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
#         [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
#         [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32]
#     ], dtype=np.uint8)
#     h, w = train_id_map.shape
#     color = np.zeros((h, w, 3), dtype=np.uint8)
#     valid = (train_id_map < len(palette))
#     color[valid] = palette[train_id_map[valid]]
#     return color  # RGB


# class SegPublisher(Node):
#     """推論結果（カラーのみ）を配信する ROS2 ノード"""
#     def __init__(self, color_topic='/segmentation/color'):
#         super().__init__('segmentation_publisher')
#         self.bridge = CvBridge()
#         self.pub_color = self.create_publisher(RosImage, color_topic, 1)

#     def publish_color(self, color_rgb_np: np.ndarray,
#                       stamp_ns: int = None, frame_id: str = 'camera'):
#         """
#         color_rgb_np: (H,W,3) uint8 (RGB)
#         """
#         # RViz2 互換の bgr8 で配信
#         msg_color = self.bridge.cv2_to_imgmsg(color_rgb_np[:, :, ::-1].copy(), encoding='bgr8')

#         if stamp_ns is not None:
#             sec = int(stamp_ns // 1_000_000_000)
#             nsec = int(stamp_ns % 1_000_000_000)
#             msg_color.header.stamp.sec = sec
#             msg_color.header.stamp.nanosec = nsec
#         else:
#             now = self.get_clock().now().to_msg()
#             msg_color.header.stamp = now

#         msg_color.header.frame_id = frame_id
#         self.pub_color.publish(msg_color)


# class Evaluator(object):
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device(args.device)

#         # image transform
#         input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
#         ])

#         # ===== Live ROS2 Dataset & DataLoader =====
#         self.val_dataset = LiveROS2SegmentationDataset(
#             img_topic='/theta_s/equirectangular/image_raw/compressed',   # ★ここを実トピックに合わせて変更
#             mask_topic=None,                # GT を配信しているならトピック名を指定
#             crop_size=(331, 1280),
#             transform=input_transform
#         )
#         # IterableDataset なので batch_sampler は使わない。ROS2 のため num_workers=0 固定
#         self.val_loader = data.DataLoader(
#             dataset=self.val_dataset,
#             batch_size=1,
#             num_workers=0,
#             pin_memory=True
#         )

#         self.classes = self.val_dataset.classes

#         # ===== Network =====
#         self.model = get_segmentation_model().to(self.device)
#         if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and cfg.MODEL.BN_EPS_FOR_ENCODER:
#             logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
#             self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

#         if args.distributed:
#             self.model = nn.parallel.DistributedDataParallel(
#                 self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
#             )
#         self.model.to(self.device)

#         self.metric = SegmentationMetric(self.val_dataset.num_class, args.distributed)

#     def set_batch_norm_attr(self, named_modules, attr, value):
#         for m in named_modules:
#             if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
#                 setattr(m[1], attr, value)

#     def visualize_segmentation(self, image, output, target, filename, save_dir="seg_results"):
#         os.makedirs(save_dir, exist_ok=True)

#         # モデル出力→trainId
#         output = output.argmax(0).cpu().numpy()  # (H, W), 0..18

#         # 画像は 0-1 正規化
#         image = image.cpu().numpy().transpose(1, 2, 0)
#         image = image - image.min()
#         image = image / max(image.max(), 1e-6)

#         # 可視化（予測のみ）
#         plt.imshow(output, cmap=Cscmap, alpha=1.0, vmin=0, vmax=18, interpolation='none')
#         plt.axis("off")
#         save_path = os.path.join(save_dir, f"{filename}_seg_only.png")
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         plt.close()

#     def eval(self):
#         # self.metric.reset()
#         self.model.eval()
#         model = self.model.module if self.args.distributed else self.model

#         logging.info("Start validation (streaming input; length unknown)")

#         # ---- ROS2 publisher 初期化（カラーのみ） ----
#         if not rclpy.ok():
#             rclpy.init(args=None)
#         seg_pub = SegPublisher(color_topic='/segmentation/color')

#         try:
#             for i, (image, target, filename) in enumerate(self.val_loader):
#                 # image: [B,C,H,W], target: [] or [B,H,W], filename: list[str] (batch_size=1)
#                 image = image.to(self.device)
#                 has_target = (isinstance(target, torch.Tensor) and target.numel() > 0)
#                 if has_target:
#                     target = target.to(self.device)

#                 # 推論
#                 t0 = time.perf_counter()
#                 with torch.no_grad():
#                     output = model.evaluate(image)  # [B,C,H,W]
#                 dt_ms = (time.perf_counter() - t0) * 1000.0
#                 print(f"Processing time for {filename}: {dt_ms:.3f} ms")

#                 # バッチ1前提
#                 out_b = output[0].cpu()
#                 pred = out_b.argmax(0).numpy().astype(np.uint8)  # (H,W) trainId
#                 color = colorize_trainid(pred)                    # (H,W,3) RGB

#                 name0 = filename[0] if isinstance(filename, (list, tuple)) else filename
#                 tgt0 = (target[0].cpu() if has_target
#                         else torch.zeros_like(torch.from_numpy(pred), dtype=torch.long))
#                 self.visualize_segmentation(image[0].cpu(), out_b, tgt0, name0)

#                 # stamp 継承（filename が "<stamp>.png" の場合）
#                 stamp_ns = None
#                 try:
#                     base = str(name0)
#                     if base.endswith('.png'):
#                         base = base[:-4]
#                     if base.isdigit():
#                         stamp_ns = int(base)
#                 except Exception:
#                     stamp_ns = None

#                 # 配信（カラーのみ）
#                 seg_pub.publish_color(color, stamp_ns=stamp_ns, frame_id='camera')

#         finally:
#             # クリーンアップ
#             try:
#                 seg_pub.destroy_node()
#             finally:
#                 if rclpy.ok():
#                     rclpy.shutdown()
#             if hasattr(self.val_loader.dataset, "close"):
#                 self.val_loader.dataset.close()


# if __name__ == '__main__':
#     # パス設定
#     cur_path = os.path.abspath(os.path.dirname(__file__))
#     root_path = os.path.split(cur_path)[0]
#     sys.path.append(root_path)

#     args = parse_args()
#     cfg.update_from_file(args.config_file)
#     cfg.update_from_list(args.opts)
#     cfg.PHASE = 'test'
#     cfg.ROOT_PATH = root_path
#     cfg.check_and_freeze()

#     default_setup(args)

#     evaluator = Evaluator(args)
#     evaluator.eval()
