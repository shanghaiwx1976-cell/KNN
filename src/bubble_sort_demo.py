"""
冒泡排序算法可视化演示程序

展示冒泡排序的工作原理，包括排序过程动画和性能分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from bubble_sort import BubbleSort, bubble_sort_simple
import time


def visualize_sorting_process(arr, steps, save_path=None):
    """
    可视化排序过程的每一步

    参数:
    ----------
    arr : list
        原始数组
    steps : list
        排序过程的步骤列表
    save_path : str, optional
        保存图片的路径
    """
    # 只显示部分关键步骤（如果步骤太多）
    max_steps = 20
    if len(steps) > max_steps:
        # 均匀采样显示关键步骤
        indices = np.linspace(0, len(steps) - 1, max_steps, dtype=int)
        display_steps = [steps[i] for i in indices]
    else:
        display_steps = steps

    n_steps = len(display_steps)
    cols = 4
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    fig.suptitle('冒泡排序过程可视化', fontsize=16, fontweight='bold')

    # 将axes转换为一维数组
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, step in enumerate(display_steps):
        ax = axes[idx]
        array_state = step['array']
        comparing = step['comparing']

        # 绘制柱状图
        colors = ['lightblue'] * len(array_state)

        # 高亮正在比较的元素
        if len(comparing) == 2:
            colors[comparing[0]] = 'orange' if not step['swapped'] else 'red'
            colors[comparing[1]] = 'orange' if not step['swapped'] else 'red'

        bars = ax.bar(range(len(array_state)), array_state, color=colors, edgecolor='black')

        # 在柱子上显示数值
        for i, (bar, val) in enumerate(zip(bars, array_state)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{val}', ha='center', va='bottom', fontsize=9)

        ax.set_ylim(0, max(arr) * 1.2)
        ax.set_xlim(-0.5, len(array_state) - 0.5)
        ax.set_xticks(range(len(array_state)))
        ax.set_title(f'步骤 {idx + 1}', fontsize=10)
        ax.set_ylabel('值', fontsize=9)

        # 添加描述
        if step['swapped']:
            ax.text(0.5, 0.95, '交换', transform=ax.transAxes,
                   ha='center', va='top', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 隐藏多余的子图
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   可视化图已保存至: {save_path}")

    return fig


def create_sorting_animation(arr, steps, save_path=None):
    """
    创建排序过程的动画

    参数:
    ----------
    arr : list
        原始数组
    steps : list
        排序过程的步骤列表
    save_path : str, optional
        保存动画的路径（.gif）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        step = steps[frame]
        array_state = step['array']
        comparing = step['comparing']

        # 绘制柱状图
        colors = ['lightblue'] * len(array_state)

        # 高亮正在比较的元素
        if len(comparing) == 2:
            colors[comparing[0]] = 'orange' if not step['swapped'] else 'red'
            colors[comparing[1]] = 'orange' if not step['swapped'] else 'red'

        bars = ax.bar(range(len(array_state)), array_state, color=colors, edgecolor='black', linewidth=2)

        # 在柱子上显示数值
        for i, (bar, val) in enumerate(zip(bars, array_state)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylim(0, max(arr) * 1.2)
        ax.set_xlim(-0.5, len(array_state) - 0.5)
        ax.set_xticks(range(len(array_state)))
        ax.set_xlabel('索引', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.set_title(f'冒泡排序动画 - {step["description"]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    anim = animation.FuncAnimation(fig, update, frames=len(steps),
                                 interval=500, repeat=True)

    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"   动画已保存至: {save_path}")
        except Exception as e:
            print(f"   警告: 无法保存动画 - {e}")

    return fig, anim


def demo_basic_sorting():
    """
    基本排序演示
    """
    print("=" * 60)
    print("冒泡排序演示 - 基本功能")
    print("=" * 60)

    # 测试数组
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n原始数组: {test_array}")

    # 创建排序器
    sorter = BubbleSort(track_steps=True)

    # 升序排序
    print("\n1. 升序排序...")
    sorted_asc = sorter.sort(test_array.copy(), ascending=True)
    print(f"   排序结果: {sorted_asc}")
    stats = sorter.get_statistics()
    print(f"   比较次数: {stats['comparisons']}")
    print(f"   交换次数: {stats['swaps']}")
    print(f"   总步骤数: {stats['steps']}")

    # 降序排序
    print("\n2. 降序排序...")
    sorter_desc = BubbleSort(track_steps=False)
    sorted_desc = sorter_desc.sort(test_array.copy(), ascending=False)
    print(f"   排序结果: {sorted_desc}")
    stats_desc = sorter_desc.get_statistics()
    print(f"   比较次数: {stats_desc['comparisons']}")
    print(f"   交换次数: {stats_desc['swaps']}")


def demo_visualization():
    """
    可视化演示
    """
    print("\n" + "=" * 60)
    print("冒泡排序演示 - 可视化")
    print("=" * 60)

    # 创建测试数组
    test_array = [45, 23, 67, 12, 89, 34, 56]
    print(f"\n原始数组: {test_array}")

    # 使用冒泡排序并记录步骤
    sorter = BubbleSort(track_steps=True)
    sorted_array = sorter.sort(test_array.copy())
    steps = sorter.get_steps()

    print(f"\n排序结果: {sorted_array}")
    print(f"总步骤数: {len(steps)}")

    # 创建可视化
    print("\n正在创建可视化...")
    visualize_sorting_process(test_array, steps,
                             save_path='/home/user/KNN/bubble_sort_steps.png')


def demo_performance_analysis():
    """
    性能分析演示
    """
    print("\n" + "=" * 60)
    print("冒泡排序演示 - 性能分析")
    print("=" * 60)

    # 测试不同大小的数组
    sizes = [10, 20, 50, 100, 200, 500]
    comparisons_list = []
    swaps_list = []
    times_list = []

    print("\n测试不同大小数组的排序性能:")
    print("\n   大小 | 比较次数 | 交换次数 | 时间(ms)")
    print("   " + "-" * 50)

    for size in sizes:
        # 生成随机数组
        arr = np.random.randint(1, 1000, size).tolist()

        # 测量时间
        sorter = BubbleSort(track_steps=False)
        start_time = time.time()
        sorter.sort(arr)
        end_time = time.time()

        stats = sorter.get_statistics()
        elapsed_ms = (end_time - start_time) * 1000

        comparisons_list.append(stats['comparisons'])
        swaps_list.append(stats['swaps'])
        times_list.append(elapsed_ms)

        print(f"   {size:4d} | {stats['comparisons']:8d} | {stats['swaps']:8d} | {elapsed_ms:8.2f}")

    # 绘制性能分析图
    print("\n正在创建性能分析图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('冒泡排序性能分析', fontsize=16, fontweight='bold')

    # 比较次数
    axes[0, 0].plot(sizes, comparisons_list, marker='o', linewidth=2,
                   markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('数组大小', fontsize=12)
    axes[0, 0].set_ylabel('比较次数', fontsize=12)
    axes[0, 0].set_title('比较次数 vs 数组大小', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 交换次数
    axes[0, 1].plot(sizes, swaps_list, marker='s', linewidth=2,
                   markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('数组大小', fontsize=12)
    axes[0, 1].set_ylabel('交换次数', fontsize=12)
    axes[0, 1].set_title('交换次数 vs 数组大小', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 执行时间
    axes[1, 0].plot(sizes, times_list, marker='^', linewidth=2,
                   markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('数组大小', fontsize=12)
    axes[1, 0].set_ylabel('时间 (ms)', fontsize=12)
    axes[1, 0].set_title('执行时间 vs 数组大小', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 时间复杂度对比（理论 vs 实际）
    theoretical = [n * n for n in sizes]  # O(n²)
    # 归一化以便比较
    theoretical_normalized = [t / theoretical[0] * comparisons_list[0] for t in theoretical]

    axes[1, 1].plot(sizes, comparisons_list, marker='o', linewidth=2,
                   markersize=8, label='实际比较次数', color='#2E86AB')
    axes[1, 1].plot(sizes, theoretical_normalized, marker='x', linewidth=2,
                   markersize=8, linestyle='--', label='理论 O(n²)', color='red')
    axes[1, 1].set_xlabel('数组大小', fontsize=12)
    axes[1, 1].set_ylabel('比较次数', fontsize=12)
    axes[1, 1].set_title('实际 vs 理论时间复杂度', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/KNN/bubble_sort_performance.png', dpi=300, bbox_inches='tight')
    print("   性能分析图已保存至: bubble_sort_performance.png")


def demo_special_cases():
    """
    特殊情况演示
    """
    print("\n" + "=" * 60)
    print("冒泡排序演示 - 特殊情况")
    print("=" * 60)

    # 已排序数组
    print("\n1. 已排序数组（最佳情况）:")
    sorted_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"   输入: {sorted_arr}")
    sorter = BubbleSort(track_steps=False)
    result = sorter.sort(sorted_arr.copy())
    stats = sorter.get_statistics()
    print(f"   输出: {result}")
    print(f"   比较次数: {stats['comparisons']}, 交换次数: {stats['swaps']}")

    # 逆序数组
    print("\n2. 逆序数组（最坏情况）:")
    reverse_arr = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    print(f"   输入: {reverse_arr}")
    sorter = BubbleSort(track_steps=False)
    result = sorter.sort(reverse_arr.copy())
    stats = sorter.get_statistics()
    print(f"   输出: {result}")
    print(f"   比较次数: {stats['comparisons']}, 交换次数: {stats['swaps']}")

    # 包含重复元素
    print("\n3. 包含重复元素:")
    duplicate_arr = [5, 2, 8, 2, 9, 1, 5, 5]
    print(f"   输入: {duplicate_arr}")
    sorter = BubbleSort(track_steps=False)
    result = sorter.sort(duplicate_arr.copy())
    stats = sorter.get_statistics()
    print(f"   输出: {result}")
    print(f"   比较次数: {stats['comparisons']}, 交换次数: {stats['swaps']}")

    # 单个元素
    print("\n4. 单个元素:")
    single_arr = [42]
    print(f"   输入: {single_arr}")
    sorter = BubbleSort(track_steps=False)
    result = sorter.sort(single_arr.copy())
    stats = sorter.get_statistics()
    print(f"   输出: {result}")
    print(f"   比较次数: {stats['comparisons']}, 交换次数: {stats['swaps']}")

    # 空数组
    print("\n5. 空数组:")
    empty_arr = []
    print(f"   输入: {empty_arr}")
    sorter = BubbleSort(track_steps=False)
    result = sorter.sort(empty_arr.copy())
    stats = sorter.get_statistics()
    print(f"   输出: {result}")
    print(f"   比较次数: {stats['comparisons']}, 交换次数: {stats['swaps']}")


def demo_step_by_step():
    """
    逐步演示排序过程
    """
    print("\n" + "=" * 60)
    print("冒泡排序演示 - 详细步骤")
    print("=" * 60)

    test_array = [5, 3, 8, 4, 2]
    print(f"\n原始数组: {test_array}")
    print("\n排序过程:")

    sorter = BubbleSort(track_steps=True)
    result = sorter.sort(test_array.copy())
    steps = sorter.get_steps()

    # 只显示关键步骤
    print("\n关键步骤:")
    for i, step in enumerate(steps):
        if '交换' in step['description'] or '完成' in step['description'] or '初始' in step['description']:
            print(f"   步骤 {i}: {step['array']} - {step['description']}")

    print(f"\n最终结果: {result}")
    print(f"统计信息: 比较 {sorter.comparisons} 次, 交换 {sorter.swaps} 次")


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_sorting()
    demo_step_by_step()
    demo_special_cases()
    demo_visualization()
    demo_performance_analysis()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)

    # 显示所有图形（如果在支持GUI的环境中）
    try:
        plt.show()
    except:
        print("\n注意: 无法显示图形界面，但图片已保存到文件中。")
