#!/usr/bin/env python3
"""
独立 YOLO 实时检测脚本
采集线程 + 推理线程，逻辑与 Mission.check_target() 一致
按 q / Ctrl-C 退出
"""
import cv2
import numpy as np
import threading
import time
from loguru import logger
from ultralytics import YOLO
from collections import defaultdict
from typing import Tuple
# "runs/train/2025_H/yolo12_more_200/weights/best_int8_openvino_model"
# ========== 参数区 ==========
MODEL_PATH = "runs/train/2025_H/last/n_back_pt/weights/best_int8_openvino_model"   # 换成你的模型
CAM_ID     = 0                                           # 摄像头号
ROI_REL    = (0.0, 0.0, 1.0, 1.0)                  # 中心裁剪比例
IMGSZ      = 320                                         # 推理尺寸
CONF_THR   = 0.7                                        # 置信度阈值
# ============================

def get_roi(img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = roi
    x = int(x * img.shape[1])
    y = int(y * img.shape[0])
    w = int(w * img.shape[1])
    h = int(h * img.shape[0])
    return img[y:y + h, x:x + w]

class YoloDetect:
    def __init__(self):
        logger.info("Loading YOLO ...")
        self.model = YOLO(MODEL_PATH, task='detect')
        self.cap = cv2.VideoCapture(CAM_ID)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")
        # 线程间共享
        self._lock = threading.Lock()
        self._img = None          # 最新 ROI
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()         # 默认放行
        self.objects = []         # 最新推理结果
        logger.success("YOLO init done")

    # ---------- 线程：采集 ----------
    def _capture_loop(self):
        logger.info("Capture thread start")
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                roi = get_roi(frame, ROI_REL)
                with self._lock:
                    self._img = roi
            time.sleep(0.01)  # 让出 CPU
        logger.info("Capture thread exit")

    # ---------- 线程：推理 ----------
    def _inference_loop(self):
        logger.info("Inference thread start")
        names = self.model.names
        CX, CY = IMGSZ // 2, IMGSZ // 2
        while not self._stop.is_set():
            with self._lock:
                if self._img is None:
                    time.sleep(0.01)
                    continue
                img = self._img.copy()
            self._pause.wait()
            results = self.model.predict(source = img,
                                        imgsz = IMGSZ,
                                        conf = CONF_THR,
                                        verbose = False,
                                        device = 'cpu',
                                        iou = 0.7,    # NMS IoU 阈值
                                        max_det = 300,
                                        )
            # 解析
            objects = []
            cls_cnt = defaultdict(int)
            for b in results[0].boxes:
                cls_cnt[names[int(b.cls)]] += 1
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls_name = names[int(b.cls)]
                objects.append({
                    'cls_id': int(b.cls),
                    'cls': cls_name,
                    'cnt': cls_cnt[cls_name],
                    'dx': cx - CX,
                    'dy': cy - CY,
                    'xyxy': (x1, y1, x2, y2)
                })
            # 原子更新
            self.objects = objects
            # 可选可视化
            vis = results[0].plot()
            cv2.imshow("YOLO-ROI", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                self._stop.set()
                break
        logger.info("Inference thread exit")

    # ---------- 启动 ----------
    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()

    # ---------- 取最新结果 ----------
    def get_objects(self):
        return self.objects

    # ---------- 暂停/继续 ----------
    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    # ---------- 退出 ----------
    def close(self):
        self._stop.set()
        self.cap.release()
        cv2.destroyAllWindows()

# ========== DEMO 主程序 ==========
if __name__ == "__main__":
    det = YoloDetect()
    det.start()
    try:
        while True:
            objs = det.get_objects()
            if objs:
                logger.info(f"Detected: {[o['cls'] for o in objs]}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        det.close()
        logger.info("Bye")