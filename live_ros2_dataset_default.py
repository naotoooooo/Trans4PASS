# # live_ros2_dataset.py
# import threading
# import time
# from collections import deque
# from typing import Optional, Tuple

# import numpy as np
# import torch
# from torch.utils.data import IterableDataset

# # ROS2
# import rclpy
# from rclpy.node import Node
# from rclpy.executors import SingleThreadedExecutor

# from sensor_msgs.msg import Image as RosImage
# from sensor_msgs.msg import CompressedImage as RosCompressedImage
# from cv_bridge import CvBridge

# from PIL import Image


# class _ImageBufferNode(Node):
#     """ROS2 Node that subscribes to image (+ optional mask) topics and buffers frames."""
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str],
#                  queue_size: int = 64):
#         super().__init__('live_image_buffer')
#         self.bridge = CvBridge()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic

#         self.img_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[BGR or mono])
#         self.msk_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[int32])
#         self.lock = threading.Lock()

#         # Image subscriber(s)
#         self.create_subscription(RosImage, img_topic, self._cb_image, 10)
#         self.create_subscription(RosCompressedImage, img_topic, self._cb_compressed, 10)
#         if mask_topic:
#             self.create_subscription(RosImage, mask_topic, self._cb_mask_image, 10)
#             self.create_subscription(RosCompressedImage, mask_topic, self._cb_mask_compressed, 10)

#     # --- Image callbacks ---
#     def _cb_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'imgmsg_to_cv2 failed: {e}')

#     def _cb_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'compressed_imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask compressed_imgmsg_to_cv2 failed: {e}')

#     # --- Retrieval helpers ---
#     def pop_synced(self, sync_tolerance_ns: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
#         """Return (img_np, mask_np or None, stamp_ns) if a synced pair (or just image) is available."""
#         with self.lock:
#             if not self.img_buf:
#                 return None
#             ti, im = self.img_buf[0]  # peek
#             if self.mask_topic is None:
#                 # Image-only mode: pop image immediately
#                 self.img_buf.popleft()
#                 return im, None, ti
#             # With mask: find nearest mask within tolerance
#             if not self.msk_buf:
#                 return None
#             # find nearest mask to ti
#             best_j = None
#             best_dt = None
#             for j, (tm, _) in enumerate(self.msk_buf):
#                 dt = abs(tm - ti)
#                 if best_dt is None or dt < best_dt:
#                     best_dt, best_j = dt, j
#                 if tm > ti and dt > (best_dt or dt):
#                     break
#             if best_j is not None and best_dt is not None and best_dt <= sync_tolerance_ns:
#                 # pop both
#                 self.img_buf.popleft()
#                 tm, mk = self.msk_buf[best_j]
#                 # remove msk_buf[best_j]
#                 self.msk_buf.rotate(-best_j)
#                 self.msk_buf.popleft()
#                 self.msk_buf.rotate(best_j)
#                 return im, mk, ti
#             return None


# class LiveROS2SegmentationDataset(IterableDataset):
#     """
#     Live (streaming) ROS2 image(+mask) dataset.
#     - Start rosbag2 playback separately (or real sensors).
#     - This dataset subscribes to topics and yields (img_tensor, mask_tensor, name).
#     Notes:
#       * Use num_workers=0 for DataLoader (ROS2 rclpy isn't fork-safe).
#       * Backpressure via internal deques (maxlen); oldest frames may drop if consumer is slow.
#     """
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str] = None,
#                  *,
#                  crop_size=(331, 1280),   # (h, w) like your validation crop
#                  sync_tolerance_ns: int = int(5e7),  # 50ms
#                  queue_size: int = 64,
#                  timeout_s: float = 5.0,
#                  to_rgb: bool = True,
#                  transform=None):
#         super().__init__()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic
#         self.crop_h, self.crop_w = crop_size
#         self.sync_tol = sync_tolerance_ns
#         self.timeout_s = timeout_s
#         self.to_rgb = to_rgb
#         self.transform = transform

#         # rclpy init (idempotent)
#         if not rclpy.ok():
#             rclpy.init(args=None)
#         self.node = _ImageBufferNode(img_topic, mask_topic, queue_size=queue_size)
#         self.executor = SingleThreadedExecutor()
#         self.executor.add_node(self.node)

#         self.spin_thread = threading.Thread(target=self._spin, daemon=True)
#         self._stop_evt = threading.Event()
#         self.spin_thread.start()

#     # --- ROS spin loop in background ---
#     def _spin(self):
#         while not self._stop_evt.is_set():
#             self.executor.spin_once(timeout_sec=0.05)

#     def close(self):
#         """Call at the end to cleanly shutdown the ROS node and executor."""
#         self._stop_evt.set()
#         try:
#             self.spin_thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.executor.remove_node(self.node)
#             self.node.destroy_node()
#         except Exception:
#             pass
#         # Only shutdown if this is the last user; leaving to caller is also okay.
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except Exception:
#             pass

#     # --- IterableDataset API ---
#     def __iter__(self):
#         # DataLoader (num_workers=0) will call this in main process
#         end_by_timeout_at = None
#         while True:
#             item = self.node.pop_synced(self.sync_tol)
#             if item is not None:
#                 im_np, mk_np, stamp = item
#                 # BGR->RGB
#                 if self.to_rgb and im_np.ndim == 3 and im_np.shape[2] == 3:
#                     im_np = im_np[:, :, ::-1]
#                 # PIL for downstream transforms (if you reuse your existing pipeline)
#                 pil_img = Image.fromarray(im_np) if im_np.ndim == 2 else Image.fromarray(im_np)
#                 pil_msk = None if mk_np is None else Image.fromarray(mk_np.astype(np.int32), mode='I')

#                 img_t, msk_t = self._apply_transforms(pil_img, pil_msk)
#                 name = f"{stamp}.png"
#                 yield img_t, (msk_t if msk_t is not None else torch.tensor([])), name
#                 end_by_timeout_at = None
#             else:
#                 # no synced item yet
#                 if end_by_timeout_at is None:
#                     end_by_timeout_at = time.time() + self.timeout_s
#                 elif time.time() > end_by_timeout_at:
#                     # Gracefully stop iteration if no data for timeout_s
#                     return
#                 time.sleep(0.01)

#     # --- same spirit as your _val_sync_transform_resize/_mask_transform ---
#     def _apply_transforms(self, pil_img: Image.Image, pil_msk: Optional[Image.Image]):
#         # ensure crop if needed
#         w, h = pil_img.size
#         cw, ch = self.crop_w, self.crop_h
#         if w < cw or h < ch:
#             pil_img = pil_img.resize((max(w, cw), max(h, ch)), Image.BILINEAR)
#             if pil_msk is not None:
#                 pil_msk = pil_msk.resize((max(w, cw), max(h, ch)), Image.NEAREST)
#             w, h = pil_img.size
#         # center-crop（ライブは安定性重視。ランダムにしたければ random に変更）
#         x1 = max(0, (w - cw) // 2)
#         y1 = max(0, (h - ch) // 2)
#         pil_img = pil_img.crop((x1, y1, x1 + cw, y1 + ch))
#         if pil_msk is not None:
#             pil_msk = pil_msk.crop((x1, y1, x1 + cw, y1 + ch))

#         # optional external transforms (e.g., torchvision)
#         if self.transform is not None:
#             img_t = self.transform(pil_img)
#         else:
#             img_t = self._default_img_transform(pil_img)

#         if pil_msk is None:
#             msk_t = None
#         else:
#             msk_np = np.array(pil_msk).astype('int32')
#             msk_t = torch.LongTensor(msk_np)
#         return img_t, msk_t

#     def _default_img_transform(self, pil_img: Image.Image) -> torch.Tensor:
#         # Minimal default: HWC uint8 -> CHW float in [0,1]
#         np_img = np.array(pil_img)
#         if np_img.ndim == 2:
#             np_img = np.expand_dims(np_img, axis=-1)
#         np_img = np.transpose(np_img, (2, 0, 1)).astype(np.float32) / 255.0
#         return torch.from_numpy(np_img)

##########################################################################################################


# # live_ros2_dataset.py
# import threading
# import time
# from collections import deque
# from typing import Optional, Tuple

# import numpy as np
# import torch
# from torch.utils.data import IterableDataset

# # ROS2
# import rclpy
# from rclpy.node import Node
# from rclpy.executors import SingleThreadedExecutor

# from sensor_msgs.msg import Image as RosImage
# from sensor_msgs.msg import CompressedImage as RosCompressedImage
# from cv_bridge import CvBridge

# from PIL import Image


# class _ImageBufferNode(Node):
#     """ROS2 Node that subscribes to image (+ optional mask) topics and buffers frames."""
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str],
#                  queue_size: int = 64):
#         super().__init__('live_image_buffer')
#         self.bridge = CvBridge()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic

#         self.img_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[BGR or mono])
#         self.msk_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[int32])
#         self.lock = threading.Lock()

#         # Image subscriber(s)
#         self.create_subscription(RosImage, img_topic, self._cb_image, 10)
#         self.create_subscription(RosCompressedImage, img_topic, self._cb_compressed, 10)
#         if mask_topic:
#             self.create_subscription(RosImage, mask_topic, self._cb_mask_image, 10)
#             self.create_subscription(RosCompressedImage, mask_topic, self._cb_mask_compressed, 10)

#     # --- Image callbacks ---
#     def _cb_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'imgmsg_to_cv2 failed: {e}')

#     def _cb_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'compressed_imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask compressed_imgmsg_to_cv2 failed: {e}')

#     # --- Retrieval helpers ---
#     def pop_synced(self, sync_tolerance_ns: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
#         """Return (img_np, mask_np or None, stamp_ns) if a synced pair (or just image) is available."""
#         with self.lock:
#             if not self.img_buf:
#                 return None
#             ti, im = self.img_buf[0]  # peek
#             if self.mask_topic is None:
#                 # Image-only mode: pop image immediately
#                 self.img_buf.popleft()
#                 return im, None, ti
#             # With mask: find nearest mask within tolerance
#             if not self.msk_buf:
#                 return None
#             # find nearest mask to ti
#             best_j = None
#             best_dt = None
#             for j, (tm, _) in enumerate(self.msk_buf):
#                 dt = abs(tm - ti)
#                 if best_dt is None or dt < best_dt:
#                     best_dt, best_j = dt, j
#                 if tm > ti and dt > (best_dt or dt):
#                     break
#             if best_j is not None and best_dt is not None and best_dt <= sync_tolerance_ns:
#                 # pop both
#                 self.img_buf.popleft()
#                 tm, mk = self.msk_buf[best_j]
#                 # remove msk_buf[best_j]
#                 self.msk_buf.rotate(-best_j)
#                 self.msk_buf.popleft()
#                 self.msk_buf.rotate(best_j)
#                 return im, mk, ti
#             return None


# class LiveROS2SegmentationDataset(IterableDataset):
#     """
#     Live (streaming) ROS2 image(+mask) dataset.
#     - Start rosbag2 playback separately (or real sensors).
#     - This dataset subscribes to topics and yields (img_tensor, mask_tensor, name).
#     Notes:
#       * Use num_workers=0 for DataLoader (ROS2 rclpy isn't fork-safe).
#       * Backpressure via internal deques (maxlen); oldest frames may drop if consumer is slow.
#     """
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str] = None,
#                  *,
#                  crop_size=(331, 1280),   # (h, w)
#                  sync_tolerance_ns: int = int(5e7),  # 50ms
#                  queue_size: int = 64,
#                  timeout_s: float = 5.0,
#                  to_rgb: bool = True,
#                  transform=None):
#         super().__init__()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic
#         self.crop_h, self.crop_w = crop_size
#         self.sync_tol = sync_tolerance_ns
#         self.timeout_s = timeout_s
#         self.to_rgb = to_rgb
#         self.transform = transform

#         # rclpy init (idempotent)
#         if not rclpy.ok():
#             rclpy.init(args=None)
#         self.node = _ImageBufferNode(img_topic, mask_topic, queue_size=queue_size)
#         self.executor = SingleThreadedExecutor()
#         self.executor.add_node(self.node)

#         self.spin_thread = threading.Thread(target=self._spin, daemon=True)
#         self._stop_evt = threading.Event()
#         self.spin_thread.start()

#         # --- Evaluator が参照するクラス情報 ---
#         self.classes = (
#             'road','sidewalk','building','wall','fence','pole','traffic light',
#             'traffic sign','vegetation','terrain','sky','person','rider','car',
#             'truck','bus','train','motorcycle','bicycle'
#         )
#         self.num_class = 19

#     @property
#     def pred_offset(self):
#         return 0

#     # --- ROS spin loop in background ---
#     def _spin(self):
#         while not self._stop_evt.is_set():
#             self.executor.spin_once(timeout_sec=0.01)

#     def close(self):
#         """Call at the end to cleanly shutdown the ROS node and executor."""
#         self._stop_evt.set()
#         try:
#             self.spin_thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.executor.remove_node(self.node)
#             self.node.destroy_node()
#         except Exception:
#             pass
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except Exception:
#             pass

#     # --- IterableDataset API ---
#     def __iter__(self):
#         # DataLoader (num_workers=0) will call this in main process
#         end_by_timeout_at = None
#         while True:
#             item = self.node.pop_synced(self.sync_tol)
#             if item is not None:
#                 im_np, mk_np, stamp = item
#                 # BGR->RGB
#                 if self.to_rgb and im_np.ndim == 3 and im_np.shape[2] == 3:
#                     im_np = im_np[:, :, ::-1]
#                 # PIL for downstream transforms (if you reuse your existing pipeline)
#                 pil_img = Image.fromarray(im_np) if im_np.ndim == 2 else Image.fromarray(im_np)
#                 pil_msk = None if mk_np is None else Image.fromarray(mk_np.astype(np.int32), mode='I')

#                 img_t, msk_t = self._apply_transforms(pil_img, pil_msk)
#                 name = f"{stamp}.png"
#                 # マスク無しは空LongTensorで返す（downstreamがdtypeで困らないように）
#                 yield img_t, (msk_t if msk_t is not None else torch.empty(0, dtype=torch.long)), name
#                 end_by_timeout_at = None
#             else:
#                 # no synced item yet
#                 if end_by_timeout_at is None:
#                     end_by_timeout_at = time.time() + self.timeout_s
#                 elif time.time() > end_by_timeout_at:
#                     # Gracefully stop iteration if no data for timeout_s
#                     return
#                 time.sleep(0.01)

#     # --- same spirit as your _val_sync_transform_resize/_mask_transform ---
#     def _apply_transforms(self, pil_img: Image.Image, pil_msk: Optional[Image.Image]):
#         # ensure crop if needed
#         w, h = pil_img.size
#         cw, ch = self.crop_w, self.crop_h
#         if w < cw or h < ch:
#             pil_img = pil_img.resize((max(w, cw), max(h, ch)), Image.BILINEAR)
#             if pil_msk is not None:
#                 pil_msk = pil_msk.resize((max(w, cw), max(h, ch)), Image.NEAREST)
#             w, h = pil_img.size
#         # center-crop（ライブは安定性重視。必要ならランダムに変更）
#         x1 = max(0, (w - cw) // 2)
#         y1 = max(0, (h - ch) // 2)
#         pil_img = pil_img.crop((x1, y1, x1 + cw, y1 + ch))
#         if pil_msk is not None:
#             pil_msk = pil_msk.crop((x1, y1, x1 + cw, y1 + ch))

#         # optional external transforms (e.g., torchvision)
#         if self.transform is not None:
#             img_t = self.transform(pil_img)
#         else:
#             img_t = self._default_img_transform(pil_img)

#         if pil_msk is None:
#             msk_t = None
#         else:
#             msk_np = np.array(pil_msk).astype('int32')
#             msk_t = torch.LongTensor(msk_np)
#         return img_t, msk_t

#     def _default_img_transform(self, pil_img: Image.Image) -> torch.Tensor:
#         # Minimal default: HWC uint8 -> CHW float in [0,1]
#         np_img = np.array(pil_img)
#         if np_img.ndim == 2:
#             np_img = np.expand_dims(np_img, axis=-1)
#         np_img = np.transpose(np_img, (2, 0, 1)).astype(np.float32) / 255.0
#         return torch.from_numpy(np_img)

#####################################################################################


# # live_ros2_dataset.py (no sync version)
# import threading
# import time
# from collections import deque
# from typing import Optional, Tuple

# import numpy as np
# import torch
# from torch.utils.data import IterableDataset

# # ROS2
# import rclpy
# from rclpy.node import Node
# from rclpy.executors import SingleThreadedExecutor

# from sensor_msgs.msg import Image as RosImage
# from sensor_msgs.msg import CompressedImage as RosCompressedImage
# from cv_bridge import CvBridge

# from PIL import Image


# class _ImageBufferNode(Node):
#     """ROS2 Node that subscribes to image (+ optional mask) topics and buffers frames.
#        - Images are popped in arrival order (FIFO).
#        - Mask, if present, is not synchronized; we just take the latest available at yield-time.
#     """
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str],
#                  queue_size: int = 1):
#         super().__init__('live_image_buffer')
#         self.bridge = CvBridge()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic

#         self.img_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[BGR or mono])
#         self.msk_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[int32])
#         self.lock = threading.Lock()

#         # Image subscriber(s)
#         self.create_subscription(RosImage, img_topic, self._cb_image, 10)
#         self.create_subscription(RosCompressedImage, img_topic, self._cb_compressed, 10)
#         if mask_topic:
#             self.create_subscription(RosImage, mask_topic, self._cb_mask_image, 10)
#             self.create_subscription(RosCompressedImage, mask_topic, self._cb_mask_compressed, 10)

#     # --- Image callbacks ---
#     def _cb_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'imgmsg_to_cv2 failed: {e}')

#     def _cb_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.img_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'compressed_imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_image(self, msg: RosImage):
#         try:
#             cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask imgmsg_to_cv2 failed: {e}')

#     def _cb_mask_compressed(self, msg: RosCompressedImage):
#         try:
#             cv = self.bridge.compressed_imgmsg_to_cv2(msg)
#             if cv.ndim == 3:
#                 cv = cv[:, :, 0]
#             cv = cv.astype(np.int32)
#             stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
#             with self.lock:
#                 self.msk_buf.append((stamp, cv))
#         except Exception as e:
#             self.get_logger().warn(f'mask compressed_imgmsg_to_cv2 failed: {e}')

#     # --- Retrieval helpers (no sync) ---
#     def pop_image(self) -> Optional[Tuple[np.ndarray, int]]:
#         """Pop the oldest image. Returns (img_np, stamp_ns) or None if no image yet."""
#         with self.lock:
#             if not self.img_buf:
#                 return None
#             ti, im = self.img_buf.popleft()
#             return im, ti

#     def latest_mask(self) -> Optional[Tuple[np.ndarray, int]]:
#         """Return the latest mask without removing it (or None if unavailable)."""
#         if self.mask_topic is None:
#             return None
#         with self.lock:
#             if not self.msk_buf:
#                 return None
#             tm, mk = self.msk_buf[-1]  # peek newest
#             return mk, tm


# class LiveROS2SegmentationDataset(IterableDataset):
#     """
#     Live (streaming) ROS2 image(+mask) dataset (NO SYNC).
#     - Images are yielded in arrival order.
#     - If mask_topic is set, the **latest** available mask at yield time is attached; otherwise None.
#     - Use num_workers=0 for DataLoader (ROS2 rclpy isn't fork-safe).
#     - Backpressure via internal deques (maxlen); oldest frames may drop if consumer is slow.
#     """
#     def __init__(self,
#                  img_topic: str,
#                  mask_topic: Optional[str] = None,
#                  *,
#                  crop_size=(331, 1280),   # (h, w)
#                  queue_size: int = 1,
#                  timeout_s: float = 5.0,
#                  to_rgb: bool = True,
#                  transform=None):
#         super().__init__()
#         self.img_topic = img_topic
#         self.mask_topic = mask_topic
#         self.crop_h, self.crop_w = crop_size
#         self.timeout_s = timeout_s
#         self.to_rgb = to_rgb
#         self.transform = transform

#         # rclpy init (idempotent)
#         if not rclpy.ok():
#             rclpy.init(args=None)
#         self.node = _ImageBufferNode(img_topic, mask_topic, queue_size=queue_size)
#         self.executor = SingleThreadedExecutor()
#         self.executor.add_node(self.node)

#         self.spin_thread = threading.Thread(target=self._spin, daemon=True)
#         self._stop_evt = threading.Event()
#         self.spin_thread.start()

#         # --- Evaluator が参照するクラス情報 ---
#         self.classes = (
#             'road','sidewalk','building','wall','fence','pole','traffic light',
#             'traffic sign','vegetation','terrain','sky','person','rider','car',
#             'truck','bus','train','motorcycle','bicycle'
#         )
#         self.num_class = 19

#     @property
#     def pred_offset(self):
#         return 0

#     # --- ROS spin loop in background ---
#     def _spin(self):
#         while not self._stop_evt.is_set():
#             self.executor.spin_once(timeout_sec=0.01)

#     def close(self):
#         """Call at the end to cleanly shutdown the ROS node and executor."""
#         self._stop_evt.set()
#         try:
#             self.spin_thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.executor.remove_node(self.node)
#             self.node.destroy_node()
#         except Exception:
#             pass
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except Exception:
#             pass

#     # --- IterableDataset API ---
#     def __iter__(self):
#         # DataLoader (num_workers=0) will call this in main process
#         end_by_timeout_at = None
#         while True:
#             popped = self.node.pop_image()
#             if popped is not None:
#                 im_np, stamp = popped
#                 # get latest mask if any (NO SYNC)
#                 lm = self.node.latest_mask()
#                 mk_np = None if lm is None else lm[0]

#                 # BGR->RGB
#                 if self.to_rgb and im_np.ndim == 3 and im_np.shape[2] == 3:
#                     im_np = im_np[:, :, ::-1]
#                 # PIL for downstream transforms
#                 pil_img = Image.fromarray(im_np) if im_np.ndim == 2 else Image.fromarray(im_np)
#                 pil_msk = None if mk_np is None else Image.fromarray(mk_np.astype(np.int32), mode='I')

#                 img_t, msk_t = self._apply_transforms(pil_img, pil_msk)
#                 name = f"{stamp}.png"
#                 yield img_t, (msk_t if msk_t is not None else torch.empty(0, dtype=torch.long)), name
#                 end_by_timeout_at = None
#             else:
#                 # no image yet
#                 if end_by_timeout_at is None:
#                     end_by_timeout_at = time.time() + self.timeout_s
#                 elif time.time() > end_by_timeout_at:
#                     # Gracefully stop iteration if no data for timeout_s
#                     return
#                 time.sleep(0.01)

#     # --- same spirit as your _val_sync_transform_resize/_mask_transform ---
#     def _apply_transforms(self, pil_img: Image.Image, pil_msk: Optional[Image.Image]):
#         # ensure crop if needed
#         w, h = pil_img.size
#         cw, ch = self.crop_w, self.crop_h
#         if w < cw or h < ch:
#             pil_img = pil_img.resize((max(w, cw), max(h, ch)), Image.BILINEAR)
#             if pil_msk is not None:
#                 pil_msk = pil_msk.resize((max(w, cw), max(h, ch)), Image.NEAREST)
#             w, h = pil_img.size
#         # center-crop
#         x1 = max(0, (w - cw) // 2)
#         y1 = max(0, (h - ch) // 2)
#         pil_img = pil_img.crop((x1, y1, x1 + cw, y1 + ch))
#         if pil_msk is not None:
#             pil_msk = pil_msk.crop((x1, y1, x1 + cw, y1 + ch))

#         # optional external transforms (e.g., torchvision)
#         if self.transform is not None:
#             img_t = self.transform(pil_img)
#         else:
#             img_t = self._default_img_transform(pil_img)

#         if pil_msk is None:
#             msk_t = None
#         else:
#             msk_np = np.array(pil_msk).astype('int32')
#             msk_t = torch.LongTensor(msk_np)
#         return img_t, msk_t

#     def _default_img_transform(self, pil_img: Image.Image) -> torch.Tensor:
#         # Minimal default: HWC uint8 -> CHW float in [0,1]
#         np_img = np.array(pil_img)
#         if np_img.ndim == 2:
#             np_img = np.expand_dims(np_img, axis=-1)
#         np_img = np.transpose(np_img, (2, 0, 1)).astype(np.float32) / 255.0
#         return torch.from_numpy(np_img)


###########################################################################################


# live_ros2_dataset.py (fixed: allocator-safe context, clean shutdown, QoS options)
# Foxy / Humble compatible

import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from cv_bridge import CvBridge

from PIL import Image


class _ImageBufferNode(Node):
    """ROS2 Node that subscribes to image (+ optional mask) topics and buffers frames.
       - Images are popped in arrival order (FIFO).
       - Mask, if present, is not synchronized; we just take the latest available at yield-time.
    """
    def __init__(self,
                 img_topic: str,
                 mask_topic: Optional[str],
                 queue_size: int = 1,
                 *,
                 context: Context,
                 reliability: ReliabilityPolicy = ReliabilityPolicy.BEST_EFFORT):
        # IMPORTANT: bind this Node to the provided Context (Foxy/Humble both support this)
        super().__init__('live_image_buffer', context=context)

        if not img_topic:
            raise ValueError("img_topic must be a non-empty string")
        if mask_topic is not None and len(mask_topic) == 0:
            raise ValueError("mask_topic must be None or a non-empty string")

        self.bridge = CvBridge()
        self.img_topic = img_topic
        self.mask_topic = mask_topic

        self.img_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[BGR or mono])
        self.msk_buf = deque(maxlen=queue_size)   # (stamp_ns, np.ndarray[int32])
        self.lock = threading.Lock()

        # QoS: default to sensor-like stream (BEST_EFFORT/volatile)
        qos = QoSProfile(
            depth=max(1, queue_size),
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Image subscriber(s)
        # Note: it's safe to subscribe to both raw and compressed; whichever publishes will feed the buffer.
        self._img_sub_raw = self.create_subscription(RosImage, img_topic, self._cb_image, qos)
        self._img_sub_cmp = self.create_subscription(RosCompressedImage, img_topic, self._cb_compressed, qos)
        if mask_topic:
            self._msk_sub_raw = self.create_subscription(RosImage, mask_topic, self._cb_mask_image, qos)
            self._msk_sub_cmp = self.create_subscription(RosCompressedImage, mask_topic, self._cb_mask_compressed, qos)

    # --- Image callbacks ---
    def _cb_image(self, msg: RosImage):
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
            with self.lock:
                self.img_buf.append((stamp, cv))
        except Exception as e:
            self.get_logger().warn(f'imgmsg_to_cv2 failed: {e}')

    def _cb_compressed(self, msg: RosCompressedImage):
        try:
            cv = self.bridge.compressed_imgmsg_to_cv2(msg)
            stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
            with self.lock:
                self.img_buf.append((stamp, cv))
        except Exception as e:
            self.get_logger().warn(f'compressed_imgmsg_to_cv2 failed: {e}')

    def _cb_mask_image(self, msg: RosImage):
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv.ndim == 3:
                cv = cv[:, :, 0]
            cv = cv.astype(np.int32)
            stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
            with self.lock:
                self.msk_buf.append((stamp, cv))
        except Exception as e:
            self.get_logger().warn(f'mask imgmsg_to_cv2 failed: {e}')

    def _cb_mask_compressed(self, msg: RosCompressedImage):
        try:
            cv = self.bridge.compressed_imgmsg_to_cv2(msg)
            if cv.ndim == 3:
                cv = cv[:, :, 0]
            cv = cv.astype(np.int32)
            stamp = (msg.header.stamp.sec * 10**9) + msg.header.stamp.nanosec
            with self.lock:
                self.msk_buf.append((stamp, cv))
        except Exception as e:
            self.get_logger().warn(f'mask compressed_imgmsg_to_cv2 failed: {e}')

    # --- Retrieval helpers (no sync) ---
    def pop_image(self) -> Optional[Tuple[np.ndarray, int]]:
        """Pop the oldest image. Returns (img_np, stamp_ns) or None if no image yet."""
        with self.lock:
            if not self.img_buf:
                return None
            ti, im = self.img_buf.popleft()
            return im, ti

    def latest_mask(self) -> Optional[Tuple[np.ndarray, int]]:
        """Return the latest mask without removing it (or None if unavailable)."""
        if self.mask_topic is None:
            return None
        with self.lock:
            if not self.msk_buf:
                return None
            tm, mk = self.msk_buf[-1]  # peek newest
            return mk, tm


class LiveROS2SegmentationDataset(IterableDataset):
    """
    Live (streaming) ROS2 image(+mask) dataset (NO SYNC) with safe shutdown.
    - Images are yielded in arrival order.
    - If mask_topic is set, the **latest** available mask at yield time is attached; otherwise None.
    - Use num_workers=0 for DataLoader (ROS2 rclpy isn't fork-safe).
    - Backpressure via internal deques (maxlen); oldest frames may drop if consumer is slow.
    """
    def __init__(self,
                 img_topic: str,
                 mask_topic: Optional[str] = None,
                 *,
                 crop_size=(331, 1280),   # (h, w)
                 queue_size: int = 1,
                 timeout_s: float = 5.0,
                 to_rgb: bool = True,
                 transform=None,
                 reliability: ReliabilityPolicy = ReliabilityPolicy.BEST_EFFORT):
        super().__init__()
        self.img_topic = img_topic
        self.mask_topic = mask_topic
        self.crop_h, self.crop_w = crop_size
        self.timeout_s = timeout_s
        self.to_rgb = to_rgb
        self.transform = transform
        self._closed = False

        # --- Dedicated Context (必ず Node/Executor と共有する) ---
        self._ctx: Context = Context()
        rclpy.init(args=None, context=self._ctx)

        # Node & Executor bound to the dedicated context
        self.node = _ImageBufferNode(img_topic, mask_topic,
                                     queue_size=queue_size,
                                     context=self._ctx,
                                     reliability=reliability)
        self.executor = SingleThreadedExecutor(context=self._ctx)
        self.executor.add_node(self.node)

        # Spin thread & stop event
        self._stop_evt = threading.Event()
        # daemon=True にしても必ず join する（終了時例外抑制）
        self._spin_thread = threading.Thread(target=self._spin, name="ros2_spin_thread", daemon=True)
        self._spin_thread.start()

        # --- Evaluator が参照するクラス情報 ---
        self.classes = (
            'road','sidewalk','building','wall','fence','pole','traffic light',
            'traffic sign','vegetation','terrain','sky','person','rider','car',
            'truck','bus','train','motorcycle','bicycle'
        )
        self.num_class = 19

    @property
    def pred_offset(self):
        return 0

    # --- ROS spin loop in background (safe) ---
    def _spin(self):
        try:
            while (not self._stop_evt.is_set()) and rclpy.ok(context=self._ctx):
                # timeout を短めにして停止フラグを素早く拾う
                self.executor.spin_once(timeout_sec=0.05)
        except ExternalShutdownException:
            # 正常な終了経路として無視
            pass
        except Exception as e:
            # 予期しないエラーはログ
            try:
                self.node.get_logger().error(f'Spin loop exception: {e}')
            except Exception:
                pass
        finally:
            # executor の shutdown は close() 側で統一して行う
            pass

    # --- Public close (call at end) ---
    def close(self):
        """Cleanly shutdown spin thread, executor, node, and context."""
        if self._closed:
            return
        self._closed = True

        # 1) 停止フラグ
        self._stop_evt.set()

        # 2) スレッド join
        try:
            self._spin_thread.join(timeout=2.0)
        except Exception:
            pass

        # 3) Executor/Node 後片付け（順序固定）
        try:
            self.executor.remove_node(self.node)
        except Exception:
            pass
        try:
            self.executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass

        # 4) Context shutdown（ok かどうかに関わらず念のため）
        try:
            rclpy.shutdown(context=self._ctx)
        except Exception:
            pass

    # 予防的に __del__ でも close を呼ぶ（ただし強制終了時は保証されない）
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # with 文で安全に使いたい場合のため
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --- IterableDataset API ---
    def __iter__(self):
        # DataLoader (num_workers=0) will call this in main process
        end_by_timeout_at = None
        while True:
            # rclpy context が終了していたら終わる
            if not rclpy.ok(context=self._ctx):
                return

            popped = self.node.pop_image()
            if popped is not None:
                im_np, stamp = popped
                # get latest mask if any (NO SYNC)
                lm = self.node.latest_mask()
                mk_np = None if lm is None else lm[0]

                # BGR->RGB
                if self.to_rgb and im_np.ndim == 3 and im_np.shape[2] == 3:
                    im_np = im_np[:, :, ::-1]
                # PIL for downstream transforms
                pil_img = Image.fromarray(im_np) if im_np.ndim == 2 else Image.fromarray(im_np)
                pil_msk = None if mk_np is None else Image.fromarray(mk_np.astype(np.int32), mode='I')

                img_t, msk_t = self._apply_transforms(pil_img, pil_msk)
                name = f"{stamp}.png"
                yield img_t, (msk_t if msk_t is not None else torch.empty(0, dtype=torch.long)), name
                end_by_timeout_at = None
            else:
                # no image yet
                if end_by_timeout_at is None:
                    end_by_timeout_at = time.time() + self.timeout_s
                elif time.time() > end_by_timeout_at:
                    # Gracefully stop iteration if no data for timeout_s
                    return
                time.sleep(0.01)

    # --- same spirit as your _val_sync_transform_resize/_mask_transform ---
    def _apply_transforms(self, pil_img: Image.Image, pil_msk: Optional[Image.Image]):
        # ensure crop if needed
        w, h = pil_img.size
        cw, ch = self.crop_w, self.crop_h
        if w < cw or h < ch:
            pil_img = pil_img.resize((max(w, cw), max(h, ch)), Image.BILINEAR)
            if pil_msk is not None:
                pil_msk = pil_msk.resize((max(w, cw), max(h, ch)), Image.NEAREST)
            w, h = pil_img.size
        # center-crop
        x1 = max(0, (w - cw) // 2)
        y1 = max(0, (h - ch) // 2)
        pil_img = pil_img.crop((x1, y1, x1 + cw, y1 + ch))
        if pil_msk is not None:
            pil_msk = pil_msk.crop((x1, y1, x1 + cw, y1 + ch))

        # optional external transforms (e.g., torchvision)
        if self.transform is not None:
            img_t = self.transform(pil_img)
        else:
            img_t = self._default_img_transform(pil_img)

        if pil_msk is None:
            msk_t = None
        else:
            msk_np = np.array(pil_msk).astype('int32')
            msk_t = torch.LongTensor(msk_np)
        return img_t, msk_t

    def _default_img_transform(self, pil_img: Image.Image) -> torch.Tensor:
        # Minimal default: HWC uint8 -> CHW float in [0,1]
        np_img = np.array(pil_img)
        if np_img.ndim == 2:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.transpose(np_img, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(np_img)

