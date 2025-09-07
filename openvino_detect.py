import cv2
import numpy as np
from openvino.runtime import Core
import yaml  # ← 添加

# ---------- 0. 加载模型 ----------
ie = Core()
model = ie.read_model("runs/train/ROBOCUP/yolo12/weights/best_openvino_model/best.xml")
compiled = ie.compile_model(model, "CPU")
input_layer = compiled.input(0)
N, C, H, W = input_layer.shape      # [1,3,320,320]
out_blob = compiled.output(0)

# 读取类别名称（支持多行 YAML）
with open("dataset/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
class_names = [str(n) for n in cfg["names"]]

# ---------- 1. 初始化摄像头 ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 0 号 USB 摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

conf_th = 0.25
nms_th  = 0.45

# ---------- 2. 预处理 ----------
def preprocess(img):
    img_rz = cv2.resize(img, (W, H))
    img_rgb = cv2.cvtColor(img_rz, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2,0,1))
    img_norm = img_chw.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, 0)

# ---------- 3. 后处理 ----------
def postprocess(frame, outs):
    h, w = frame.shape[:2]
    outs = outs[0].T                     # [8400,84]
    scores = np.max(outs[:, 4:], axis=1)
    keep = scores > conf_th
    outs = outs[keep]
    scores = scores[keep]
    cls_ids = np.argmax(outs[:, 4:], axis=1)
    boxes = outs[:, :4]                  # cx,cy,w,h
    boxes[:, [0,2]] *= w / W             # 还原到原图坐标
    boxes[:, [1,3]] *= h / H
    boxes[:, :2] -= boxes[:, 2:] / 2     # 转 xyxy
    boxes[:, 2:] += boxes[:, :2]

    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_th, nms_th)
    idxs = idxs.flatten() if len(idxs) else []  # 已保证是 list
    results = []
    for i in idxs:                             # 直接遍历
        x1, y1, x2, y2 = boxes[i].astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cls_name = class_names[cls_ids[i]]
        results.append((cls_name, (cx, cy), (x1, y1, x2, y2)))
    return results

# ---------- 4. 主循环 ----------
while True:
    t0 = cv2.getTickCount()          # <<< 计时开始

    ret, frame = cap.read()
    if not ret:
        break
    blob = preprocess(frame)
    preds = compiled([blob])[out_blob]
    detections = postprocess(frame, preds)

    # 显示结果
    for name, (cx, cy), (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {cx},{cy}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # <<< 计时结束 + 打印
    latency = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
    print(f"{latency:.1f} ms")

    cv2.imshow("YOLOv8 USB Cam (INT8)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()