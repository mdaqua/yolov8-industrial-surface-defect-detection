# 实验记录

## EXP-001 YOLOv8n Baseline

- Model: yolov8n.pt
- Dataset: 待填写
- Image Size: 640
- Epochs: 100
- Batch Size: 16
- Result:
  - Precision:
  - Recall:
  - mAP@0.5:
  - mAP@0.5:0.95:
- Analysis:
  - 小目标缺陷可能存在漏检，需要重点检查 Recall。
  - scratches 类别可能与背景纹理混淆，需要关注误检样本。
- Next:
  - 补充负样本和小缺陷样本。
  - 对比 YOLOv8s。
  - 根据业务目标调节 conf 和 iou 阈值。
