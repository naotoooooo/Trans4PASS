# tools/eval_dp.py
from __future__ import print_function

import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms

# Segmentron / Trans4PASS
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup

# === ROS2 publish 用（カラーのみ配信） ===
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage

# 可視化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ライブ入力 Dataset
from live_ros2_dataset import LiveROS2SegmentationDataset


# ------------------------------------------------------------
# カラーマップ定義（Cityscapes trainId=0..18）
# ------------------------------------------------------------
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
], dtype=float)
Cscolor[:, :3] /= 256.0
Cscmap = ListedColormap(Cscolor)


def colorize_trainid(train_id_map: np.ndarray) -> np.ndarray:
    """(H,W) uint8 の trainId マップを (H,W,3) uint8 (RGB) に変換"""
    palette = np.array([
        [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
        [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
        [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32]
    ], dtype=np.uint8)
    h, w = train_id_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (train_id_map < len(palette))
    color[valid] = palette[train_id_map[valid]]
    return color  # RGB


class SegPublisher(Node):
    """推論結果（カラーのみ）を配信する ROS2 ノード"""
    def __init__(self, color_topic='/segmentation/color'):
        super().__init__('segmentation_publisher')
        self.bridge = CvBridge()
        self.pub_color = self.create_publisher(RosImage, color_topic, 1)

    def publish_color(self, color_rgb_np: np.ndarray,
                      stamp=None,
                      frame_id: str = None):
        """
        color_rgb_np: (H,W,3) uint8 (RGB)
        stamp: builtin_interfaces.msg.Time or None
        frame_id: str or None
        """
        # RViz2 互換の bgr8 で配信
        msg_color = self.bridge.cv2_to_imgmsg(color_rgb_np[:, :, ::-1].copy(), encoding='bgr8')

        # タイムスタンプを元画像に合わせる
        if stamp is not None:
            msg_color.header.stamp = stamp
        else:
            msg_color.header.stamp = self.get_clock().now().to_msg()

        # frame_id も元画像に合わせる
        if frame_id is not None:
            msg_color.header.frame_id = frame_id
        else:
            msg_color.header.frame_id = 'camera'

        self.pub_color.publish(msg_color)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # ===== Live ROS2 Dataset & DataLoader =====
        self.val_dataset = LiveROS2SegmentationDataset(
            img_topic='/equirectangular/image/compressed',   # ★ここを実トピックに合わせて変更
            mask_topic=None,                # GT を配信しているならトピック名を指定
            crop_size=(331, 1280),
            transform=input_transform
        )
        # IterableDataset なので batch_sampler は使わない。ROS2 のため num_workers=0 固定
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=self._collate_noop,
        )

        self.classes = self.val_dataset.classes

        # ===== Network =====
        self.model = get_segmentation_model().to(self.device)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )
        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def visualize_segmentation(self, image, output, target, filename, save_dir="seg_results"):
        os.makedirs(save_dir, exist_ok=True)

        # モデル出力→trainId
        output = output.argmax(0).cpu().numpy()  # (H, W), 0..18

        # 画像は 0-1 正規化
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image - image.min()
        image = image / max(image.max(), 1e-6)

        # 可視化（予測のみ）
        plt.imshow(output, cmap=Cscmap, alpha=1.0, vmin=0, vmax=18, interpolation='none')
        plt.axis("off")
        save_path = os.path.join(save_dir, f"{filename}_seg_only.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    @staticmethod
    def _collate_noop(batch):
        """DataLoader から返ってきたものをそのまま返す (batch_size=1 前提)"""
        return batch

    def eval(self):
        # self.metric.reset()
        self.model.eval()
        model = self.model.module if self.args.distributed else self.model

        logging.info("Start validation (streaming input; length unknown)")

        # ---- ROS2 publisher 初期化（カラーのみ） ----
        if not rclpy.ok():
            rclpy.init(args=None)
        seg_pub = SegPublisher(color_topic='/segmentation/color')

        # try:
        #     # Dataset からは (image, target, stamp, frame_id) が返ってくる想定
        #     for i, (image, target, stamp, frame_id) in enumerate(self.val_loader):
        #         # image: [B,C,H,W]
        #         # target: [] or [B,H,W]
        #         # stamp: [Time] or Time
        #         # frame_id: [str] or str

        #         image = image.to(self.device)
        #         has_target = (isinstance(target, torch.Tensor) and target.numel() > 0)
        #         if has_target:
        #             target = target.to(self.device)

        #         # 推論
        #         t0 = time.perf_counter()
        #         with torch.no_grad():
        #             output = model.evaluate(image)  # [B,C,H,W]
        #         dt_ms = (time.perf_counter() - t0) * 1000.0
        #         print(f"Processing time: {dt_ms:.3f} ms")

        #         # バッチ1前提
        #         out_b = output[0].cpu()
        #         pred = out_b.argmax(0).numpy().astype(np.uint8)  # (H,W) trainId
        #         color = colorize_trainid(pred)                    # (H,W,3) RGB

        #         # target が無い場合はダミー
        #         tgt0 = (target[0].cpu() if has_target
        #                 else torch.zeros_like(torch.from_numpy(pred), dtype=torch.long))

        #         # stamp/frame_id は DataLoader によって list 包装されるので剥がす
        #         stamp0 = stamp[0] if isinstance(stamp, (list, tuple)) else stamp
        #         frame_id0 = frame_id[0] if isinstance(frame_id, (list, tuple)) else frame_id

        #         # 可視化用のファイル名（stamp から ns を作る）
        #         try:
        #             stamp_ns = stamp0.sec * 1_000_000_000 + stamp0.nanosec
        #             name0 = f"{stamp_ns}"
        #         except Exception:
        #             name0 = f"frame_{i:06d}"

        #         self.visualize_segmentation(image[0].cpu(), out_b, tgt0, name0)

        #         # 配信（カラーのみ）: 元画像の stamp / frame_id をそのまま引き継ぐ
        #         seg_pub.publish_color(color, stamp=stamp0, frame_id=frame_id0)
        try:
            # DataLoader は collate_fn=noop なので、各イテレーションで
            #   batch = [ (image, target, stamp, frame_id), ... ]
            # が返ってくる。batch_size=1 なので batch[0] だけ見る。
            for i, batch in enumerate(self.val_loader):
                image, target, stamp, frame_id = batch[0]
                # image: [C,H,W] (単一サンプル)
                # target: [] or [H,W]
                # stamp: builtin_interfaces.msg.Time
                # frame_id: str

                # ここから下は少しだけ調整
                image = image.unsqueeze(0).to(self.device)  # [1,C,H,W] にする
                has_target = (isinstance(target, torch.Tensor) and target.numel() > 0)
                if has_target:
                    target = target.unsqueeze(0).to(self.device)  # [1,H,W] に揃える

                # 推論
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model.evaluate(image)  # [B,C,H,W] = [1,C,H,W]
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(f"Processing time: {dt_ms:.3f} ms")

                out_b = output[0].cpu()  # [C,H,W]
                pred = out_b.argmax(0).numpy().astype(np.uint8)
                color = colorize_trainid(pred)

                tgt0 = (target[0].cpu() if has_target
                        else torch.zeros_like(torch.from_numpy(pred), dtype=torch.long))

                # 可視化用ファイル名（stamp から ns）
                try:
                    stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                    name0 = f"{stamp_ns}"
                except Exception:
                    name0 = f"frame_{i:06d}"

                self.visualize_segmentation(image[0].cpu(), out_b, tgt0, name0)

                # 元画像の stamp / frame_id をそのまま使って publish
                seg_pub.publish_color(color, stamp=stamp, frame_id=frame_id)

        finally:
            # クリーンアップ
            try:
                seg_pub.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()
            if hasattr(self.val_loader.dataset, "close"):
                self.val_loader.dataset.close()


if __name__ == '__main__':
    # パス設定
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(cur_path)[0]
    sys.path.append(root_path)

    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
