import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO('runs/train/2025_H/yolo12/weights/best_int8_openvino_model', task='detect')
    results = model.predict(
        source=0,
        imgsz=320,
        stream=True,
        show=False,
        conf=0.4,
        verbose=True,
    )
    # 提前把类别名称读出来，后面好映射 id → name
    names = model.names          # 例如 {0: 'ball', 1: 'goal', ...}
    for result in results:
        frame = result.orig_img
        annotated = result.plot()
        # ---------------- 关键部分 ----------------
        # 遍历当前帧里所有检测框
        for box in result.boxes:
            # 1. 坐标 (x1, y1, x2, y2) 是左上、右下两个角点
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 如果想拿中心点
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # 2. 置信度
            conf = float(box.conf[0])
            # 3. 类别 id 和名称
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            # 这里你可以把数据存进列表、做逻辑判断、打印等等
            print(f'{cls_id+1}, conf:{conf:.2f}')
        # ----------------------------------------
        cv2.imshow('YOLOv8 Real-time', annotated)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()