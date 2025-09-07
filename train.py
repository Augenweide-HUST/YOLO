import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

yaml = 'yolo12.yaml'

if __name__ == '__main__':
    # yolov12n back pt
    model = YOLO(f'ultralytics/cfg/models/12/{yaml}')
    model.load('yolov12n.pt')  # 初始化时可能需要指定从哪个权重文件加载
    model.train(
        data='dataset/data_back.yaml',
        cache=False,
        imgsz=320,
        epochs=200,
        batch=16,
        close_mosaic=0,
        workers=8,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
        optimizer='SGD',  # 使用 SGD 优化器
        patience=50,  #  earlystop
        # resume=True,  # 断点续训，YOLO 初始化时选择 last.pt
        # amp=False,  # 关闭 amp，loss 出现 nan 时可以尝试
        # fraction=0.2,
        project='runs/train/2025_H/last',
        name=yaml.replace('yolo12.yaml', 'n_back_pt'),
    )
    model = YOLO("runs/train/2025_H/last/n_back_pt/weights/best.pt")
    model.export(
        format="openvino",     # Intel 优化格式
        imgsz=320,             # 更小输入尺寸，减少计算量
        simplify=True,         # 图结构简化
        dynamic=False,         # 固定输入尺寸，避免额外开销
        optimize=True,          # 面向 CPU 的图优化
        int8=True,              # INT8 量化，体积减半
        data="dataset/data_back.yaml"
    )
    # yolov12n pt
    model = YOLO(f'ultralytics/cfg/models/12/{yaml}')
    model.load('yolov12n.pt')  # 初始化时可能需要指定从哪个权重文件加载
    model.train(
        data='dataset/data.yaml',
        cache=False,
        imgsz=320,
        epochs=200,
        batch=16,
        close_mosaic=0,
        workers=8,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
        optimizer='SGD',  # 使用 SGD 优化器
        patience=50,  #  earlystop
        # resume=True,  # 断点续训，YOLO 初始化时选择 last.pt
        # amp=False,  # 关闭 amp，loss 出现 nan 时可以尝试
        # fraction=0.2,
        project='runs/train/2025_H/last',
        name=yaml.replace('yolo12.yaml', 'n_pt'),
    )
    model = YOLO("runs/train/2025_H/last/n_pt/weights/best.pt")
    model.export(
        format="openvino",     # Intel 优化格式
        imgsz=320,             # 更小输入尺寸，减少计算量
        simplify=True,         # 图结构简化
        dynamic=False,         # 固定输入尺寸，避免额外开销
        optimize=True,          # 面向 CPU 的图优化
        int8=True,              # INT8 量化，体积减半
        data="dataset/data.yaml"
    )