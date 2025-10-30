"""
K-最近邻（K-Nearest Neighbors）算法实现

这是一个从零开始实现的KNN分类器，支持多种距离度量方法。
"""

import numpy as np
from collections import Counter


class KNN:
    """
    K-最近邻分类器

    参数:
    ----------
    k : int, 默认=3
        用于投票的最近邻数量
    distance_metric : str, 默认='euclidean'
        距离度量方法，可选: 'euclidean', 'manhattan'
    """

    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练KNN模型（实际上只是存储训练数据）

        参数:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            训练特征
        y : numpy.ndarray, shape (n_samples,)
            训练标签
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _calculate_distance(self, x1, x2):
        """
        计算两个向量之间的距离

        参数:
        ----------
        x1 : numpy.ndarray
            向量1
        x2 : numpy.ndarray
            向量2

        返回:
        ----------
        float
            两个向量之间的距离
        """
        if self.distance_metric == 'euclidean':
            # 欧氏距离: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # 曼哈顿距离: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"不支持的距离度量方法: {self.distance_metric}")

    def _predict_single(self, x):
        """
        对单个样本进行预测

        参数:
        ----------
        x : numpy.ndarray
            单个测试样本

        返回:
        ----------
        预测的类别标签
        """
        # 计算测试样本与所有训练样本的距离
        distances = []
        for x_train in self.X_train:
            dist = self._calculate_distance(x, x_train)
            distances.append(dist)

        # 找出k个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个最近邻的标签
        k_nearest_labels = self.y_train[k_indices]

        # 通过投票决定预测结果（选择出现次数最多的类别）
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        对测试数据进行预测

        参数:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            测试特征

        返回:
        ----------
        numpy.ndarray, shape (n_samples,)
            预测的类别标签
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        X = np.array(X)
        predictions = []

        # 对每个测试样本进行预测
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X, y):
        """
        计算模型在给定数据上的准确率

        参数:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            测试特征
        y : numpy.ndarray, shape (n_samples,)
            真实标签

        返回:
        ----------
        float
            准确率（0到1之间）
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def main():
    """
    简单的使用示例
    """
    # 创建一个简单的数据集
    X_train = np.array([
        [1, 2], [2, 3], [3, 3],  # 类别 0
        [6, 6], [7, 8], [8, 8]   # 类别 1
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # 测试数据
    X_test = np.array([[2, 2], [7, 7], [1, 3]])
    y_test = np.array([0, 1, 0])

    # 创建并训练KNN分类器
    knn = KNN(k=3, distance_metric='euclidean')
    knn.fit(X_train, y_train)

    # 预测
    predictions = knn.predict(X_test)
    print(f"预测结果: {predictions}")
    print(f"真实标签: {y_test}")

    # 计算准确率
    accuracy = knn.score(X_test, y_test)
    print(f"准确率: {accuracy:.2%}")


if __name__ == "__main__":
    main()
