"""
冒泡排序算法实现

冒泡排序是一种简单的排序算法，它重复地遍历要排序的数组，
一次比较两个元素，如果它们的顺序错误就把它们交换过来。
"""


class BubbleSort:
    """
    冒泡排序实现类

    提供基本冒泡排序和优化版本的冒泡排序
    """

    def __init__(self, track_steps=False):
        """
        初始化冒泡排序器

        参数:
        ----------
        track_steps : bool, default=False
            是否记录排序过程中的每一步
        """
        self.track_steps = track_steps
        self.steps = []  # 记录排序过程
        self.comparisons = 0  # 比较次数
        self.swaps = 0  # 交换次数

    def sort(self, arr, ascending=True):
        """
        对数组进行冒泡排序

        参数:
        ----------
        arr : list
            待排序的数组
        ascending : bool, default=True
            True为升序，False为降序

        返回:
        ----------
        list
            排序后的数组
        """
        # 重置统计信息
        self.comparisons = 0
        self.swaps = 0
        self.steps = []

        # 复制数组以避免修改原数组
        result = arr.copy()
        n = len(result)

        # 记录初始状态
        if self.track_steps:
            self.steps.append({
                'array': result.copy(),
                'comparing': [],
                'swapped': False,
                'description': '初始状态'
            })

        # 冒泡排序主循环
        for i in range(n):
            # 提前退出标志（优化）
            swapped = False

            # 内层循环，进行相邻元素比较
            for j in range(0, n - i - 1):
                self.comparisons += 1

                # 根据升序或降序决定是否交换
                should_swap = (result[j] > result[j + 1]) if ascending else (result[j] < result[j + 1])

                if should_swap:
                    # 交换元素
                    result[j], result[j + 1] = result[j + 1], result[j]
                    self.swaps += 1
                    swapped = True

                    # 记录交换步骤
                    if self.track_steps:
                        self.steps.append({
                            'array': result.copy(),
                            'comparing': [j, j + 1],
                            'swapped': True,
                            'description': f'交换位置 {j} 和 {j + 1}: {result[j + 1]} ↔ {result[j]}'
                        })
                else:
                    # 记录比较但未交换的步骤
                    if self.track_steps:
                        self.steps.append({
                            'array': result.copy(),
                            'comparing': [j, j + 1],
                            'swapped': False,
                            'description': f'比较位置 {j} 和 {j + 1}: 无需交换'
                        })

            # 如果这一轮没有发生交换，说明已经排序完成
            if not swapped:
                if self.track_steps:
                    self.steps.append({
                        'array': result.copy(),
                        'comparing': [],
                        'swapped': False,
                        'description': f'第 {i + 1} 轮：未发生交换，排序完成'
                    })
                break
            else:
                if self.track_steps:
                    self.steps.append({
                        'array': result.copy(),
                        'comparing': [],
                        'swapped': False,
                        'description': f'第 {i + 1} 轮完成'
                    })

        return result

    def get_statistics(self):
        """
        获取排序统计信息

        返回:
        ----------
        dict
            包含比较次数和交换次数的字典
        """
        return {
            'comparisons': self.comparisons,
            'swaps': self.swaps,
            'steps': len(self.steps)
        }

    def get_steps(self):
        """
        获取排序过程的所有步骤

        返回:
        ----------
        list
            排序过程中的每一步
        """
        return self.steps


def bubble_sort_simple(arr, ascending=True):
    """
    简单的冒泡排序函数（不记录过程）

    参数:
    ----------
    arr : list
        待排序的数组
    ascending : bool, default=True
        True为升序，False为降序

    返回:
    ----------
    list
        排序后的数组
    """
    sorter = BubbleSort(track_steps=False)
    return sorter.sort(arr, ascending)


def bubble_sort_with_steps(arr, ascending=True):
    """
    带步骤记录的冒泡排序函数

    参数:
    ----------
    arr : list
        待排序的数组
    ascending : bool, default=True
        True为升序，False为降序

    返回:
    ----------
    tuple
        (排序后的数组, 排序器对象)
    """
    sorter = BubbleSort(track_steps=True)
    result = sorter.sort(arr, ascending)
    return result, sorter
