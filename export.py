from ultralytics import YOLO

model = YOLO("runs/train/2025_exp/yolo12_more/weights/best.pt")

model.export(
    format="openvino",     # Intel 优化格式
    imgsz=320,             # 更小输入尺寸，减少计算量
    simplify=True,         # 图结构简化
    dynamic=False,         # 固定输入尺寸，避免额外开销
    optimize=True,          # 面向 CPU 的图优化
    int8=True,              # INT8 量化，体积减半
    data="dataset/data.yaml"
)

