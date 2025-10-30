# KNN算法演示程序

K-最近邻（K-Nearest Neighbors）算法的Python实现和可视化演示。

## 功能特性

- 从零实现的KNN分类算法
- 支持多种距离度量方法（欧氏距离、曼哈顿距离）
- 可视化决策边界
- 使用经典鸢尾花数据集进行演示
- 简洁易懂的代码实现

## 项目结构

```
KNN/
├── src/
│   ├── knn.py              # KNN算法核心实现
│   └── demo.py             # 演示程序
├── README.md               # 项目说明
└── requirements.txt        # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行演示程序

```bash
python src/demo.py
```

演示程序会：
1. 加载鸢尾花数据集
2. 使用KNN算法进行分类
3. 显示分类准确率
4. 绘制决策边界可视化图

### 基本使用示例

```python
from src.knn import KNN
import numpy as np

# 准备训练数据
X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# 创建KNN分类器
knn = KNN(k=3)

# 训练（存储训练数据）
knn.fit(X_train, y_train)

# 预测
X_test = np.array([[2, 2], [7, 7]])
predictions = knn.predict(X_test)

print(predictions)  # 输出: [0 1]
```

## KNN算法原理

K-最近邻算法是一种简单但有效的分类算法：

1. **训练阶段**：存储所有训练数据
2. **预测阶段**：
   - 计算待预测样本与所有训练样本的距离
   - 找出距离最近的K个样本
   - 通过投票决定分类结果（K个邻居中出现最多的类别）

### 参数说明

- `k`: 邻居数量，通常选择奇数以避免平票
- `distance_metric`: 距离度量方法
  - `'euclidean'`: 欧氏距离（默认）
  - `'manhattan'`: 曼哈顿距离

## 示例输出

运行演示程序后，你将看到：
- 在测试集上的准确率
- 决策边界可视化图（不同颜色区域代表不同类别）

## 许可证

MIT License

## 作者

Generated with Claude Code
