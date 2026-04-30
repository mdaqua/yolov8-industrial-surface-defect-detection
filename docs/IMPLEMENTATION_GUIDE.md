# 基于 YOLOv8 的工业表面缺陷检测项目实现原理说明

这份文档面向刚接触视觉算法和目标检测工程的读者。它不会只告诉你“运行哪条命令”，而是尽量解释：这个项目为什么要这样组织目录，YOLO 标签为什么长成这种格式，训练、验证、推理、API 服务之间是什么关系，以及每个 Python 模块在整条链路中负责什么。

读完后，你应该能回答三个问题：

- 一张工业表面图片如何变成 YOLOv8 能学习的数据。
- YOLOv8 训练完成后，模型如何输出缺陷类别、位置框和置信度。
- 为什么一个算法实验项目还需要配置文件、日志、API、Docker 和测试。

## 1. 这个项目解决什么问题

工业表面缺陷检测的输入是一张产品表面图片，例如钢材、布匹、金属零件或电池片表面。输出不是简单判断“有缺陷/无缺陷”，而是更细的目标检测结果：

```json
{
  "class_name": "scratches",
  "confidence": 0.91,
  "bbox": {
    "x1": 120.5,
    "y1": 80.3,
    "x2": 230.7,
    "y2": 118.9
  }
}
```

这表示模型认为图片里有一个 `scratches` 类型缺陷，置信度是 `0.91`，缺陷位置由左上角和右下角坐标描述。

本项目的核心是把这个任务做成一条完整工程链路：

```text
原始图片和标签
  -> 数据检查和数据划分
  -> YOLOv8 训练
  -> 验证集指标评估
  -> 单图或批量推理
  -> FastAPI HTTP 服务
  -> Docker 部署
```

对初学者来说，这条链路比单独训练一个模型更重要。真实项目里，模型只是中间一环；数据、配置、结果保存、接口和部署同样决定项目能不能复现和交付。

## 2. 项目目录如何理解

项目目录按“职责”拆分，而不是把所有代码写进一个脚本：

```text
.
├── configs/          # 配置：训练参数、服务参数
├── data/             # 数据配置和数据目录
├── src/              # 核心代码
│   ├── dataset/      # 数据检查、划分、data.yaml 生成
│   ├── training/     # 训练入口
│   ├── evaluation/   # 验证和指标导出
│   ├── inference/    # 推理封装
│   ├── api/          # HTTP 服务
│   └── utils/        # 通用工具
├── tests/            # 测试
├── weights/          # 模型权重，例如 best.pt
├── outputs/          # 上传图片和推理可视化结果
└── runs/             # 训练、验证日志和结果
```

这种结构背后的原则是单一职责：

- `dataset` 只关心数据是否正确，不关心模型怎么训练。
- `training` 只关心如何调用 YOLOv8 训练。
- `evaluation` 只关心训练后模型表现如何。
- `inference` 只关心如何把模型输出转成业务可读 JSON。
- `api` 只关心 HTTP 请求和响应。

这样做的好处是：以后你想换数据集、换模型、加前端、加相机输入，都不需要把整个项目推倒重写。

## 3. 目标检测里最重要的数据概念

### 3.1 图片和标签必须一一对应

YOLO 目标检测数据通常是一张图片对应一个同名 `.txt` 标签文件：

```text
images/a.jpg
labels/a.txt

images/b.jpg
labels/b.txt
```

代码中 [check_image_label_pairs](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:12) 做的就是这件事：

- 找出有图片但没有标签的样本。
- 找出有标签但没有图片的孤立标签。
- 统计正常匹配的样本数量。

为什么要先检查？因为训练阶段如果路径或标签不对，模型可能直接报错，也可能悄悄跳过大量样本，最后得到一个看似训练成功但实际无效的模型。

### 3.2 YOLO 标签格式

YOLO 检测标签每一行代表一个目标框：

```text
class_id x_center y_center width height
```

例如：

```text
5 0.5123 0.4871 0.2134 0.0952
```

含义是：

- `5`：类别编号，比如 `scratches`。
- `0.5123`：框中心点横坐标，占整张图片宽度的比例。
- `0.4871`：框中心点纵坐标，占整张图片高度的比例。
- `0.2134`：框宽度，占整张图片宽度的比例。
- `0.0952`：框高度，占整张图片高度的比例。

注意，这里不是像素坐标，而是归一化坐标，范围必须在 `0` 到 `1` 之间。

如果原图宽度是 `1000`，某个缺陷框中心点横坐标是 `512` 像素，那么归一化后的 `x_center` 是：

```text
512 / 1000 = 0.512
```

YOLO 使用归一化坐标的原因是：不同图片尺寸可能不一样，归一化后标签能和图片尺寸解耦。

### 3.3 空标签不是错误

工业场景里经常会有“无缺陷图片”，它们没有任何目标框。YOLO 允许空标签文件：

```text
images/good_001.jpg
labels/good_001.txt    # 文件存在，但内容为空
```

这类负样本很重要。它们能告诉模型：不是所有纹理、反光、边缘都应该被当成缺陷。负样本太少时，模型容易误检。

项目中的 [split_dataset](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:73) 支持 `require_labels=False`。命令行里对应的是：

```bash
python -m src.dataset.prepare ... --allow-negative
```

开启后，缺少标签文件的图片会被当作无缺陷样本，并生成空标签文件。

## 4. 数据处理模块如何工作

数据处理模块位于：

```text
src/dataset/prepare.py
```

它主要提供五个能力。

### 4.1 检查图片和标签配对

[check_image_label_pairs](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:12) 使用文件名的 `stem` 做匹配。

`a.jpg` 的 stem 是 `a`，`a.txt` 的 stem 也是 `a`，所以它们是一对。

输出结构大致是：

```python
{
    "image_count": 1000,
    "label_count": 1000,
    "matched_count": 998,
    "missing_labels": ["bad_003"],
    "missing_images": ["old_label_001"]
}
```

### 4.2 校验标签是否合法

[validate_yolo_label](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:35) 和 [collect_label_errors](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:40) 会检查：

- 每一行是不是 5 个字段。
- `class_id` 是否为整数。
- `class_id` 是否超出类别数量。
- 坐标是否都是数字。
- 坐标是否在 `0` 到 `1` 之间。

这些检查看起来基础，但很关键。目标检测模型对标签质量非常敏感，错误框、越界框、错类别都会让模型学习到错误模式。

### 4.3 划分训练集、验证集、测试集

常见比例是：

```text
train: 70%
val:   20%
test:  10%
```

三者作用不同：

- `train`：模型真正用于学习的数据。
- `val`：训练过程中用来观察模型泛化能力的数据。
- `test`：最终评估模型时使用，尽量不要参与调参。

[split_dataset](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:73) 使用 `seed` 固定随机划分。固定随机种子的意义是可复现：同一份数据、同一个 seed，每次划分结果都一样。

### 4.4 生成 YOLOv8 需要的 data.yaml

YOLOv8 训练时需要知道三个信息：

- 数据集根目录在哪里。
- 训练、验证、测试图片分别在哪。
- 类别编号和类别名称怎么对应。

对应的配置文件是：

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled_in_scale
  5: scratches
```

项目中的默认文件是 [data/data.yaml](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/data/data.yaml:1)。

### 4.5 统计类别分布

[class_distribution](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/dataset/prepare.py:149) 会统计每个类别有多少个标注框。

如果结果类似：

```python
{0: 500, 1: 480, 2: 12, 3: 430, 4: 460, 5: 450}
```

说明类别 `2` 样本明显偏少。类别不平衡会导致模型更容易学好大类，对小类召回较差。

## 5. YOLOv8 训练入口如何理解

训练代码位于：

```text
src/training/train.py
```

核心函数是 [train_model](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/training/train.py:12)。

它做了四件事：

1. 读取 `configs/train.yaml`。
2. 根据命令行参数覆盖部分配置。
3. 用 `YOLO(model_name)` 加载预训练模型。
4. 调用 `model.train(**config)` 启动训练。

配置文件 [configs/train.yaml](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/configs/train.yaml:1) 中的关键参数如下：

```yaml
model: yolov8n.pt
data: data/data.yaml
epochs: 100
imgsz: 640
batch: 16
device: cpu
project: runs/train
name: defect_yolov8n
```

这些参数可以这样理解：

- `model`：初始模型。`yolov8n.pt` 小而快，适合跑通流程；`yolov8s.pt` 更大，通常精度更好。
- `data`：数据集配置文件，告诉 YOLO 去哪里找图片和标签。
- `epochs`：完整看完训练集多少轮。
- `imgsz`：输入模型前图片缩放到多大。小缺陷任务可以尝试提高到 `960`。
- `batch`：一次送入模型多少张图片。显存不足时需要降低。
- `device`：用 `cpu`、`0` 或其他 GPU 编号。
- `project` 和 `name`：训练结果保存在哪里。

训练完成后，Ultralytics 通常会保存：

```text
runs/train/defect_yolov8n/weights/best.pt
runs/train/defect_yolov8n/weights/last.pt
```

- `best.pt`：验证集表现最好的权重。
- `last.pt`：最后一个 epoch 的权重。

部署和推理时一般使用 `best.pt`。

## 6. 验证和指标如何理解

验证代码位于：

```text
src/evaluation/validate.py
```

核心函数是 [validate_model](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/evaluation/validate.py:13)。它加载训练好的权重，然后调用：

```python
metrics = model.val(...)
```

最终输出：

```python
{
    "precision": ...,
    "recall": ...,
    "map50": ...,
    "map50_95": ...,
    "class_map50_95": [...]
}
```

初学者可以先这样理解这些指标：

- `Precision`：模型报出来的缺陷里，有多少是真的。低 Precision 说明误检多。
- `Recall`：真实缺陷里，有多少被模型找到了。低 Recall 说明漏检多。
- `mAP@0.5`：IoU 阈值为 `0.5` 时的综合检测能力。
- `mAP@0.5:0.95`：多个 IoU 阈值下的平均表现，比 `mAP@0.5` 更严格。

### 6.1 IoU 是什么

IoU 衡量预测框和真实框重叠程度：

```text
IoU = 预测框和真实框的交集面积 / 预测框和真实框的并集面积
```

如果预测框和真实框几乎完全重合，IoU 接近 `1`。如果两个框没重叠，IoU 是 `0`。

### 6.2 工业检测更关注哪个指标

这取决于业务成本：

- 漏检成本高：优先关注 Recall，例如缺陷品流出会造成严重后果。
- 误检成本高：优先关注 Precision，例如误杀合格品会造成大量返工。

工业缺陷检测通常更怕漏检，所以 Recall 很重要。但 Precision 太低也会让产线频繁误报警。因此最终常常需要调节 `conf` 阈值，在误检和漏检之间取平衡。

## 7. 推理模块如何把模型输出变成 JSON

推理封装位于：

```text
src/inference/detector.py
```

核心类是 [DefectDetector](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/inference/detector.py:14)。

### 7.1 初始化模型

初始化时会检查权重文件是否存在：

```python
self.model_path = Path(model_path)
if not self.model_path.exists():
    raise FileNotFoundError(...)
```

这样设计是为了让错误尽早暴露。如果没有 `weights/best.pt`，服务不会假装自己能推理。

然后加载模型：

```python
self.model = YOLO(str(self.model_path))
```

### 7.2 执行预测

[predict](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/inference/detector.py:33) 会调用：

```python
results = self.model.predict(
    source=str(image),
    conf=conf,
    iou=iou,
    imgsz=self.imgsz,
    device=self.device,
    save=False,
    verbose=False,
)
```

几个参数很重要：

- `source`：输入图片路径。
- `conf`：置信度阈值。低于这个阈值的检测框会被过滤掉。
- `iou`：NMS 去重时使用的 IoU 阈值。
- `imgsz`：推理时图片输入尺寸。
- `save=False`：不让 Ultralytics 自动保存，由项目自己控制保存位置。

### 7.3 conf 阈值怎么影响结果

`conf` 是置信度阈值。

- 调低 `conf`：模型会报出更多框，Recall 可能提高，但误检也可能增加。
- 调高 `conf`：结果更保守，Precision 可能提高，但漏检也可能增加。

工业场景里，如果先追求“不漏缺陷”，可以从较低阈值开始，比如 `0.2` 或 `0.25`；如果误报太多，再逐步提高。

### 7.4 iou 阈值和 NMS

目标检测模型可能对同一个缺陷预测出多个相近框。NMS 的作用是保留最可信的框，去掉重复框。

`iou` 控制“多像才算重复”：

- 阈值较低：更容易删除相邻框，可能误删靠得很近的多个缺陷。
- 阈值较高：更宽松，可能保留更多重复框。

默认 `0.7` 是一个相对常见的起点，具体要看数据。

### 7.5 解析 Ultralytics 输出

Ultralytics 的原始输出不是直接给业务接口用的 JSON。项目通过 [parse_results](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/inference/detector.py:66) 进行转换。

对每个检测框，提取：

```python
class_id = int(box.cls[0])
confidence = float(box.conf[0])
xyxy = box.xyxy[0].tolist()
```

然后整理为：

```json
{
  "class_id": 5,
  "class_name": "scratches",
  "confidence": 0.912346,
  "bbox": {
    "x1": 120.457,
    "y1": 80.333,
    "x2": 230.789,
    "y2": 118.944
  }
}
```

这里的 `xyxy` 是像素坐标：

- `x1, y1`：左上角。
- `x2, y2`：右下角。

注意它和训练标签中的 `x_center y_center width height` 不一样。训练标签用归一化中心点格式，推理输出给用户时更适合用像素级 `xyxy` 格式，因为更直观，也方便画框。

## 8. 命令行推理脚本如何工作

命令行入口位于：

```text
src/inference/predict.py
```

它读取 [configs/app.yaml](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/configs/app.yaml:1)，创建 `DefectDetector`，然后对单张图片或文件夹里的多张图片执行推理。

单图示例：

```bash
python -m src.inference.predict --image path/to/image.jpg --model weights/best.pt --save
```

文件夹示例：

```bash
python -m src.inference.predict --image path/to/images --model weights/best.pt --save
```

`--save` 会保存可视化结果图片。可视化保存由 [save_visualization](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/inference/detector.py:96) 完成，它使用 Ultralytics 的 `result.plot()` 把检测框画到图片上。

## 9. FastAPI 服务如何封装模型

API 服务位于：

```text
src/api/main.py
```

它提供两个接口：

```text
GET  /health
POST /predict
```

### 9.1 /health 的意义

[health_check](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/api/main.py:32) 返回：

```json
{
  "status": "ok",
  "model_path": "weights/best.pt",
  "model_available": false,
  "model_loaded": false
}
```

即使模型还没准备好，健康检查也能访问。这对部署很有用：你可以区分“服务进程是否活着”和“模型权重是否已经就位”。

### 9.2 为什么使用 lru_cache 加载模型

[get_detector](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/api/main.py:23) 上有：

```python
@lru_cache(maxsize=1)
```

这意味着模型只会加载一次。第一次 `/predict` 请求进来时加载模型，后续请求复用同一个模型对象。

为什么不每次请求都加载模型？因为模型权重可能几十到上百 MB，每次加载都会很慢，也浪费内存。

### 9.3 /predict 的处理流程

[predict](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/src/api/main.py:43) 的流程是：

```text
检查上传文件是不是图片
  -> 检查文件大小
  -> 用 PIL 验证图片内容是否可读
  -> 保存上传图片到 outputs/uploads
  -> 调用 DefectDetector.predict
  -> 保存可视化结果到 outputs/results
  -> 返回 JSON
```

这里做了几层异常处理：

- 非图片：返回 `400`。
- 图片过大：返回 `413`。
- 图片损坏：返回 `400`。
- 模型不存在或推理失败：返回 `500`。

这体现了算法工程和普通脚本的区别：接口面对的是不确定输入，不能默认用户永远上传正确文件。

## 10. Docker 部署如何理解

Docker 的目标是让项目在不同机器上拥有尽量一致的运行环境。

[Dockerfile](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/Dockerfile:1) 做了这些事：

1. 使用 `python:3.10-slim` 作为基础镜像。
2. 安装 OpenCV 运行时需要的系统库。
3. 安装 `requirements.txt` 中的 Python 依赖。
4. 复制项目代码。
5. 用 `uvicorn` 启动 FastAPI 服务。

[docker-compose.yml](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/docker-compose.yml:1) 额外配置了端口和挂载目录：

```yaml
volumes:
  - ./weights:/app/weights
  - ./outputs:/app/outputs
  - ./configs:/app/configs
```

这样做的好处是：

- 权重可以在容器外替换。
- 推理结果保存在宿主机，容器重启后不丢。
- 配置可以在宿主机修改，不必每次重新构建镜像。

## 11. 测试在这个项目里测什么

测试目录是：

```text
tests/
```

当前测试重点不是训练一个真实模型，因为那需要数据、GPU 和较长时间。测试覆盖的是更稳定、更适合自动化检查的部分：

- YOLO 标签格式是否合法。
- 数据划分数量是否符合预期。
- 模型输出解析后的 JSON 结构是否正确。
- `/health` 接口是否能返回服务状态。

这是一种实用策略：耗时长、依赖重的训练实验手动跑；纯逻辑、容易回归的部分用单元测试守住。

## 12. 初学者如何按顺序跑通项目

建议按下面顺序来，不要一开始就上 Docker 或大模型。

### 第一步：准备环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 第二步：准备数据

把图片和标签放成类似结构：

```text
data/raw/images/
data/raw/labels/
```

然后运行：

```bash
python -m src.dataset.prepare \
  --image-dir data/raw/images \
  --label-dir data/raw/labels \
  --output-dir data/processed \
  --classes crazing inclusion patches pitted_surface rolled_in_scale scratches
```

### 第三步：训练 baseline

先用小模型：

```bash
python -m src.training.train --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

如果能跑通，再考虑更大的模型或更高输入分辨率。

### 第四步：验证模型

```bash
python -m src.evaluation.validate \
  --model runs/train/defect_yolov8n/weights/best.pt \
  --data data/data.yaml
```

观察 Precision、Recall、mAP，以及 Ultralytics 生成的混淆矩阵和 PR 曲线。

### 第五步：准备推理权重

把最佳权重放到统一位置：

```text
weights/best.pt
```

### 第六步：单图推理

```bash
python -m src.inference.predict --image path/to/image.jpg --model weights/best.pt --save
```

### 第七步：启动 API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

访问：

```bash
curl http://localhost:8000/health
```

上传图片：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

## 13. 常见问题和排查思路

### 13.1 训练时报找不到图片或标签

优先检查：

- `data/data.yaml` 中的 `path` 是否正确。
- `images/train` 和 `labels/train` 是否都存在。
- 图片名和标签名是否同名。

### 13.2 训练能跑，但指标很差

常见原因：

- 标签框不准。
- 类别编号和 `names` 对不上。
- 类别分布严重不均衡。
- 缺陷太小，`imgsz=640` 不够。
- 训练轮数太少。
- 数据里有大量模糊、反光或重复纹理干扰。

### 13.3 误检很多

可以尝试：

- 增加无缺陷负样本。
- 提高 `confidence_threshold`。
- 检查背景纹理是否和缺陷太像。
- 增加相似背景但无缺陷的图片。

### 13.4 漏检很多

可以尝试：

- 降低 `confidence_threshold`。
- 提高输入尺寸，比如 `imgsz=960`。
- 使用 `yolov8s.pt` 或更大模型。
- 补充小缺陷样本。
- 检查是否有漏标。

### 13.5 API 返回模型不存在

确认文件存在：

```text
weights/best.pt
```

如果训练结果在：

```text
runs/train/defect_yolov8n/weights/best.pt
```

需要复制到 `weights/best.pt`，或修改 [configs/app.yaml](/Users/mdaqua/Project/yolov8-industrial-surface-defect-detection/configs/app.yaml:1) 中的 `model_path`。

## 14. 后续可以如何扩展

这个项目是一个 baseline 工程，后续可以沿三个方向扩展。

数据层面：

- 增加负样本。
- 对小缺陷做裁剪或切片。
- 统一标注规范。
- 统计每类样本数量，针对少数类补数据。

模型层面：

- 对比 `yolov8n.pt`、`yolov8s.pt`、`yolov8m.pt`。
- 尝试更大 `imgsz`。
- 根据误检漏检调整 `conf` 和 `iou`。
- 如果需要精细轮廓，尝试分割模型。

工程层面：

- 增加批量 API。
- 增加前端可视化页面。
- 导出 ONNX 或 TensorRT 加速推理。
- 加入相机或视频流输入。
- 增加接口鉴权和请求日志分析。

## 15. 一句话总结

这个项目的实现重点不是“把 YOLOv8 调起来”这么简单，而是把工业缺陷检测做成一条可复现、可检查、可部署的工程链路。对于视觉算法初学者，最值得掌握的是：数据格式要严谨，训练参数要可配置，验证指标要会解释，推理输出要能被业务系统使用，服务部署要能稳定处理异常输入。
