# KNN算法使用示例

## 示例1: 基本使用

```python
from src.knn import KNN
import numpy as np

# 准备数据
X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[2, 2], [7, 7]])

# 创建并训练模型
knn = KNN(k=3)
knn.fit(X_train, y_train)

# 预测
predictions = knn.predict(X_test)
print(f"预测结果: {predictions}")  # 输出: [0 1]
```

## 示例2: 使用曼哈顿距离

```python
from src.knn import KNN
import numpy as np

# 创建数据
X_train = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# 使用曼哈顿距离
knn = KNN(k=3, distance_metric='manhattan')
knn.fit(X_train, y_train)

# 预测并评估
X_test = np.array([[1, 0], [6, 5]])
y_test = np.array([0, 1])

predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

print(f"预测结果: {predictions}")
print(f"准确率: {accuracy:.2%}")
```

## 示例3: 多分类问题

```python
from src.knn import KNN
import numpy as np

# 创建三类数据
X_train = np.array([
    [1, 1], [1, 2], [2, 1],        # 类别 0
    [5, 5], [5, 6], [6, 5],        # 类别 1
    [9, 9], [9, 10], [10, 9]       # 类别 2
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 测试数据
X_test = np.array([[1.5, 1.5], [5.5, 5.5], [9.5, 9.5]])
y_test = np.array([0, 1, 2])

# 训练和预测
knn = KNN(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

print(f"预测结果: {predictions}")
print(f"准确率: {accuracy:.2%}")
```

## 示例4: 选择最佳K值

```python
from src.knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 测试不同的K值
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracies.append(accuracy)
    print(f"K={k}: 准确率={accuracy:.4f}")

# 找到最佳K值
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\n最佳K值: {best_k}, 最高准确率: {best_accuracy:.4f}")
```

## 示例5: 与sklearn的KNN对比

```python
from src.knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 我们的实现
our_knn = KNN(k=5)
our_knn.fit(X_train, y_train)
our_accuracy = our_knn.score(X_test, y_test)

# sklearn的实现
sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)

print(f"我们的KNN准确率: {our_accuracy:.4f}")
print(f"sklearn KNN准确率: {sklearn_accuracy:.4f}")
print(f"差异: {abs(our_accuracy - sklearn_accuracy):.4f}")
```

## 示例6: 处理不平衡数据

```python
from src.knn import KNN
import numpy as np

# 创建不平衡数据集（类别0有30个样本，类别1有10个样本）
np.random.seed(42)
X_class0 = np.random.randn(30, 2) + np.array([2, 2])
X_class1 = np.random.randn(10, 2) + np.array([6, 6])

X_train = np.vstack([X_class0, X_class1])
y_train = np.hstack([np.zeros(30), np.ones(10)])

# 创建测试集
X_test_class0 = np.random.randn(10, 2) + np.array([2, 2])
X_test_class1 = np.random.randn(10, 2) + np.array([6, 6])
X_test = np.vstack([X_test_class0, X_test_class1])
y_test = np.hstack([np.zeros(10), np.ones(10)])

# 训练模型
knn = KNN(k=5)
knn.fit(X_train, y_train)

# 评估
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

print(f"总体准确率: {accuracy:.2%}")

# 计算每个类别的准确率
for label in [0, 1]:
    mask = y_test == label
    class_accuracy = np.mean(predictions[mask] == y_test[mask])
    print(f"类别{int(label)}准确率: {class_accuracy:.2%}")
```

## 示例7: 特征标准化的重要性

```python
from src.knn import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 不标准化
knn_no_scaling = KNN(k=5)
knn_no_scaling.fit(X_train, y_train)
accuracy_no_scaling = knn_no_scaling.score(X_test, y_test)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_with_scaling = KNN(k=5)
knn_with_scaling.fit(X_train_scaled, y_train)
accuracy_with_scaling = knn_with_scaling.score(X_test_scaled, y_test)

print(f"不标准化准确率: {accuracy_no_scaling:.4f}")
print(f"标准化后准确率: {accuracy_with_scaling:.4f}")
print(f"提升: {(accuracy_with_scaling - accuracy_no_scaling):.4f}")
```

## 性能考虑

KNN算法的时间复杂度：
- 训练时间: O(1) - 只是存储数据
- 预测时间: O(n × d × m) - 其中n是训练样本数，d是特征维度，m是测试样本数

对于大规模数据集，可以考虑：
1. 使用更高效的数据结构（如KD树）
2. 减少特征维度（PCA等降维方法）
3. 使用近似最近邻算法
