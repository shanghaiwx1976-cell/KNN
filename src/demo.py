"""
KNN算法可视化演示程序

使用鸢尾花数据集演示KNN分类算法，并绘制决策边界
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import KNN


def plot_decision_boundaries(X, y, classifier, resolution=0.02):
    """
    绘制决策边界

    参数:
    ----------
    X : numpy.ndarray, shape (n_samples, 2)
        特征数据（只使用两个特征用于可视化）
    y : numpy.ndarray, shape (n_samples,)
        标签
    classifier : KNN
        训练好的KNN分类器
    resolution : float
        网格分辨率
    """
    # 设置颜色映射
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 创建网格
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 预测网格上每个点的类别
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # 绘制决策边界
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker='o',
            edgecolor='black',
            label=f'类别 {cl}',
            s=80
        )


def demo_iris():
    """
    使用鸢尾花数据集演示KNN算法
    """
    print("=" * 60)
    print("KNN算法演示 - 鸢尾花数据集分类")
    print("=" * 60)

    # 加载鸢尾花数据集
    print("\n1. 加载数据集...")
    iris = load_iris()
    # 只使用前两个特征用于可视化
    X = iris.data[:, :2]
    y = iris.target
    print(f"   数据集大小: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    print(f"   类别数量: {len(np.unique(y))}")

    # 划分训练集和测试集
    print("\n2. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   训练集: {X_train.shape[0]} 个样本")
    print(f"   测试集: {X_test.shape[0]} 个样本")

    # 特征标准化
    print("\n3. 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 测试不同的K值
    print("\n4. 测试不同的K值...")
    k_values = [1, 3, 5, 7, 9]
    best_k = None
    best_accuracy = 0

    for k in k_values:
        knn = KNN(k=k, distance_metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        print(f"   K={k}: 准确率 = {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"\n   最佳K值: {best_k}, 准确率: {best_accuracy:.2%}")

    # 使用最佳K值创建最终模型
    print(f"\n5. 使用K={best_k}训练最终模型...")
    final_knn = KNN(k=best_k, distance_metric='euclidean')
    final_knn.fit(X_train_scaled, y_train)

    # 在测试集上评估
    predictions = final_knn.predict(X_test_scaled)
    accuracy = final_knn.score(X_test_scaled, y_test)
    print(f"   测试集准确率: {accuracy:.2%}")

    # 显示一些预测示例
    print("\n6. 预测示例（前5个测试样本）:")
    print("   索引 | 真实标签 | 预测标签 | 结果")
    print("   " + "-" * 40)
    for i in range(min(5, len(y_test))):
        result = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"   {i:4d} | {y_test[i]:8d} | {predictions[i]:8d} | {result}")

    # 绘制决策边界
    print("\n7. 绘制决策边界可视化...")
    plt.figure(figsize=(12, 5))

    # 绘制训练集决策边界
    plt.subplot(1, 2, 1)
    plot_decision_boundaries(X_train_scaled, y_train, final_knn)
    plt.xlabel('特征 1 (标准化)', fontsize=12)
    plt.ylabel('特征 2 (标准化)', fontsize=12)
    plt.title(f'训练集决策边界 (K={best_k})', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # 绘制测试集决策边界
    plt.subplot(1, 2, 2)
    plot_decision_boundaries(X_test_scaled, y_test, final_knn)
    plt.xlabel('特征 1 (标准化)', fontsize=12)
    plt.ylabel('特征 2 (标准化)', fontsize=12)
    plt.title(f'测试集决策边界 (K={best_k})', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/KNN/knn_decision_boundaries.png', dpi=300, bbox_inches='tight')
    print("   可视化图已保存至: knn_decision_boundaries.png")

    # 绘制K值与准确率的关系
    print("\n8. 绘制K值与准确率的关系...")
    plt.figure(figsize=(10, 6))

    accuracies = []
    for k in k_values:
        knn = KNN(k=k, distance_metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        accuracies.append(accuracy)

    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=10,
             color='#2E86AB', markerfacecolor='#A23B72')
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('KNN算法: K值对准确率的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # 标记最佳K值
    max_idx = np.argmax(accuracies)
    plt.scatter(k_values[max_idx], accuracies[max_idx],
                color='red', s=200, zorder=5, marker='*',
                label=f'最佳K={k_values[max_idx]}')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('/home/user/KNN/k_value_analysis.png', dpi=300, bbox_inches='tight')
    print("   分析图已保存至: k_value_analysis.png")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def demo_simple():
    """
    简单的二维数据演示
    """
    print("\n" + "=" * 60)
    print("KNN算法演示 - 简单二维数据")
    print("=" * 60)

    # 创建简单的二维数据集
    np.random.seed(42)

    # 类别0: 左下角
    X_class0 = np.random.randn(30, 2) + np.array([2, 2])

    # 类别1: 右上角
    X_class1 = np.random.randn(30, 2) + np.array([6, 6])

    # 合并数据
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(30), np.ones(30)])

    # 创建KNN分类器
    knn = KNN(k=5, distance_metric='euclidean')
    knn.fit(X, y)

    # 绘制决策边界
    plt.figure(figsize=(8, 6))
    plot_decision_boundaries(X, y, knn)
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.title('KNN算法 - 简单二维数据分类 (K=5)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/user/KNN/simple_demo.png', dpi=300, bbox_inches='tight')
    print("\n可视化图已保存至: simple_demo.png")

    print("=" * 60)


if __name__ == "__main__":
    # 运行鸢尾花数据集演示
    demo_iris()

    # 运行简单数据演示
    demo_simple()

    # 显示所有图形（如果在支持GUI的环境中）
    try:
        plt.show()
    except:
        print("\n注意: 无法显示图形界面，但图片已保存到文件中。")
