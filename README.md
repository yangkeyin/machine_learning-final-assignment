# UAV Detection Project

这是一个基于深度学习的无人机检测项目，使用 EfficientNet-B0 作为基础模型来区分无人机和背景图像。

## 任务背景：
1.实用性强的现实场景：无人机在各个领域的应用越来越广泛，尤其是在安防、物流和农业领域。这一作业模拟了实际场景中的无人机监测需求，帮助学生了解图像分析在现实生活中的应用价值。
2.计算机视觉与深度学习的应用：这个题目涉及通过图像识别技术来检测无人机，反映了计算机视觉和深度学习算法在物体识别中的重要性。学生可以学习如何利用卷积神经网络（CNN）等技术来实现自动图像分类。

## 数据集简介
![image](https://github.com/user-attachments/assets/695614b8-3f6a-492d-b0e0-ad6866353b81)

## 项目结构

```
UAV-dataset/
├── dataset1/
│   ├── train/
│   │   ├── UAV/
│   │   └── background/
│   ├── val/
│   │   ├── UAV/
│   │   └── background/
│   └── test/
│       ├── UAV/
│       └── background/
├── baseline.py
├── test.py
└── results/
    ├── confusion_matrix_[timestamp].png
    ├── confusion_matrix_[timestamp].csv
    ├── classification_report_[timestamp].csv
    └── predictions_[timestamp].csv
```

## 主要特性

- 使用预训练的 EfficientNet-B0 模型
- 实现了部分层的冻结训练
- 使用数据增强提高模型鲁棒性
- 实现了早停机制和学习率调度
- 支持 MPS (Metal Performance Shaders) 加速
- 详细的测试结果分析和可视化

## 环境要求

- Python 3.10+
- PyTorch
- torchvision
- Pillow
- matplotlib
- seaborn
- pandas
- scikit-learn

## 模型架构

- 基础模型：EfficientNet-B0
- 自定义分类器：
  - Dropout(0.3)
  - Linear(in_features -> 512)
  - ReLU
  - Dropout(0.2)
  - Linear(512 -> 2)

## 训练特点

- 使用 AdamW 优化器
- 余弦退火学习率调度
- L2 正则化
- 梯度裁剪
- 早停机制
- 数据增强：
  - 随机水平翻转
  - 随机旋转
  - 颜色抖动
  - 标准化

## 使用方法

### 训练模型

```bash
python baseline.py
```

### 测试模型

```bash
python test.py
```

## 测试结果

测试结果将保存在 `results` 目录下，包括：
- 混淆矩阵可视化
- 分类报告
- 预测结果详情
- 错误预测分析

## 模型保存

训练好的模型将保存为 `best_model_efficientnet.pth`，包含：
- 模型类型
- 模型状态字典
- 训练轮次
- 最佳验证准确率

## 注意事项

1. 确保数据集按照指定的目录结构组织
2. 测试时建议使用 CPU 设备以避免 MPS 相关问题
3. 如果使用 GPU/MPS 加速，请确保相关硬件和驱动支持

## 性能优化建议

1. 调整批量大小适应内存
2. 根据具体任务调整学习率和训练轮次
3. 可以尝试调整数据增强参数
4. 考虑使用不同的模型架构或优化策略



