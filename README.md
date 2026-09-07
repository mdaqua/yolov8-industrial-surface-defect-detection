# Industrial Surface Defect Detection with YOLOv8

基于 YOLOv8 的工业表面缺陷检测实验工程，覆盖数据集整理、训练、验证、图片推理、FastAPI 服务和 Docker 部署。

## 项目结构

```text
.
├── configs/                # 训练和服务配置
├── data/                   # 数据集配置与数据目录
├── src/
│   ├── dataset/            # 数据检查、划分、data.yaml 生成
│   ├── training/           # YOLOv8 训练入口
│   ├── evaluation/         # 验证与指标导出
│   ├── inference/          # 模型推理封装
│   ├── api/                # FastAPI 服务
│   └── utils/              # 配置、日志、图片工具
├── tests/                  # 单元测试
├── weights/                # best.pt 等模型权重
├── outputs/                # 推理上传与可视化结果
├── runs/                   # 训练、验证与日志输出
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 环境安装

建议使用 Python 3.10。当前代码兼容 Python 3.9+，Docker 镜像使用 Python 3.10。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 训练时，请按服务器 CUDA 版本安装匹配的 PyTorch，再安装本项目依赖。

## 数据准备

原始数据需要包含图片目录和同名 YOLO 标签目录。标签格式为：

```text
class_id x_center y_center width height
```

示例数据划分命令：

```bash
python -m src.dataset.prepare \
  --image-dir data/raw/images \
  --label-dir data/raw/labels \
  --output-dir data/processed \
  --classes crazing inclusion patches pitted_surface rolled_in_scale scratches
```

如果包含无缺陷负样本且没有标签文件，可增加 `--allow-negative`，程序会为空样本生成空标签。

## 模型训练

训练参数在 `configs/train.yaml` 中维护。

```bash
python -m src.training.train --config configs/train.yaml
```

常用覆盖参数：

```bash
python -m src.training.train --model yolov8s.pt --epochs 150 --imgsz 960 --batch 8 --device 0
```

训练完成后，将最佳权重复制或软链接到：

```text
weights/best.pt
```

## 模型验证

```bash
python -m src.evaluation.validate \
  --model weights/best.pt \
  --data data/data.yaml \
  --split val \
  --output runs/val/metrics.json
```

输出包含 Precision、Recall、mAP@0.5 和 mAP@0.5:0.95。Ultralytics 也会在 `runs/val/` 下保存混淆矩阵、PR 曲线等图表。

## 图片推理

```bash
python -m src.inference.predict --image path/to/image.jpg --model weights/best.pt --save
```

`--image` 也可以传入图片文件夹。推理返回 JSON，包含类别、置信度和 `xyxy` 检测框；可视化图片保存到 `outputs/results/`。

## API 服务

启动服务：

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

图片检测：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

如果 `weights/best.pt` 不存在，`/health` 仍可访问，`/predict` 会返回明确的模型缺失错误。

## Docker 部署

```bash
docker compose up -d --build
```

容器会挂载本地 `weights/`、`outputs/` 和 `configs/`。部署前请确认 `weights/best.pt` 已存在。

## 测试

```bash
python -m pytest
```

测试主要覆盖 YOLO 标签校验、数据划分、推理结果解析和 API 基础行为。真实训练和真实模型推理需要准备数据集和权重后手动执行。

## 误检漏检分析方向

- Recall 低：检查小目标、模糊缺陷、输入尺寸、类别不均衡和漏标问题。
- Precision 低：检查背景纹理、反光、负样本不足和置信度阈值。
- 类别混淆：查看混淆矩阵和失败样本，统一标注标准并补充相似类别样本。

实验记录可写入 `experiments.md`。
