
---

### 📚 算法库全景分类速查

#### 1. 必知必会：排序与二分 (Sorting & Binary Search)
**前提**：二分系列必须针对**有序**序列。

| 函数名 | 描述 | 复杂度 | 典型应用场景 |
| :--- | :--- | :--- | :--- |
| `sort` | 快速排序(混合)。默认升序。 | $O(N \log N)$ | 绝大多数排序需求。 |
| `stable_sort` | **稳定**排序。相等元素的相对顺序不变。 | $O(N \log N)$ | 多关键字排序（如先按成绩排，再按名字排）。 |
| `partial_sort` | 部分排序。只排前 K 个元素。 | $O(N \log K)$ | 获取 Top K，但不需要后续有序。 |
| **`nth_element`** | **快速选择**。将第 n 小的元素放到第 n 个位置，且左小右大。 | **$O(N)$** | **求中位数、求第 K 大元素 (面试神器)。** |
| `lower_bound` | 找第一个 $\ge$ val 的位置。 | $O(\log N)$ | 查找插入位置、范围左边界。 |
| `upper_bound` | 找第一个 $>$ val 的位置。 | $O(\log N)$ | 查找范围右边界。 |
| `binary_search` | 检查元素是否存在 (返回 bool)。 | $O(\log N)$ | 简单的存在性判断。 |

#### 2. 查找与遍历 (Searching & Traversal)
**注意**：这些都是线性查找，复杂度通常为 $O(N)$。

| 函数名                             | 描述                    | 示例用法                                                     |
| :------------------------------ | :-------------------- | :------------------------------------------------------- |
| `find`                          | 查找等于 val 的第一个元素。      | `find(v.begin(), v.end(), 5)`                            |
| `find_if`                       | 查找**满足条件**的第一个元素。     | `find_if(v.begin(), v.end(), [](int x){return x%2==0;})` |
| `count`                         | 统计等于 val 的个数。         | `count(v.begin(), v.end(), 1)`                           |
| `count_if`                      | 统计**满足条件**的个数。        | `count_if(...)`                                          |
| `all_of` / `any_of` / `none_of` | 检查范围是否 全满足/任一满足/全不满足。 | `if (all_of(v.begin(), v.end(), isPositive)) ...`        |
| `mismatch`                      | 找两个序列第一个不匹配的位置。       | 比较两个字符串/数组哪里开始不同。                                        |

#### 3. 修改与搬运 (Modification)
这一类函数直接修改容器内容。

| 函数名                      | 描述                        | 核心场景                              |
| :----------------------- | :------------------------ | :-------------------------------- |
| `reverse`                | 反转范围内的元素。                 | 翻转字符串、链表转换。                       |
| **`rotate`**             | **旋转**序列。将 `middle` 移到开头。 | **数组循环移动** (例如 `k` 次轮转)。          |
| `fill`                   | 将范围填充为指定值。                | 初始化 dp 数组 `fill(dp, dp+n, -1)`。   |
| `transform`              | 对每个元素执行操作并写入新位置。          | 类似 Python 的 `map`，如将 vector 全部平方。 |
| `unique`                 | **去重** (只移除相邻重复)，返回新尾部。   | 配合 `sort` 和 `erase` 实现真正去重。       |
| `remove` / `remove_if`   | **移除**元素 (移到后面)，返回新尾部。    | **Erase-Remove Idiom** (见下文详解)。   |
| `replace` / `replace_if` | 将特定值替换为新值。                | 字符串替换空格为 `%20`。                   |

#### 4. 最值与堆操作 (Min/Max & Heap)
除了基础最值，STL 还允许你把 vector 当作堆 (Heap) 来用。

| 函数名 | 描述 | 备注 |
| :--- | :--- | :--- |
| `max` / `min` | 返回两者或列表中的最值。 | `max({1, 2, 3, 4})` |
| `max_element` | 返回最大值的**迭代器**。 | 获取数组最大值下标用 `it - v.begin()`。 |
| `min_element` | 返回最小值的**迭代器**。 | 同上。 |
| `clamp` (C++17) | 将值限制在 `[min, max]` 区间内。 | `clamp(val, 0, 100)` 防止越界。 |
| `make_heap` | 在 $O(N)$ 时间内将 vector 构建成堆。 | 默认是大顶堆。 |
| `push_heap` / `pop_heap` | 堆的插入/删除调整操作。 | 必须配合 vector 的 `push_back`/`pop_back` 使用。 |

#### 5. 集合操作 (Set Operations)
**前提**：两个序列都必须**已排序**。

| 函数名 | 描述 | 场景 |
| :--- | :--- | :--- |
| `set_intersection` | 求交集。 | 两个有序数组找公共元素。 |
| `set_union` | 求并集。 | 合并两个有序数组。 |
| `set_difference` | 求差集 (A - B)。 | 在 A 中但不在 B 中的元素。 |
| `includes` | 判断 B 是否为 A 的子序列。 | 集合包含关系判断。 |

#### 6. 排列与数学 (Permutations & Numeric)
数学相关通常在 `<numeric>` 头文件中。

| 函数名 | 头文件 | 描述 |
| :--- | :--- | :--- |
| `next_permutation` | `<algorithm>` | 下一个字典序排列 (如 123 -> 132)。 |
| `prev_permutation` | `<algorithm>` | 上一个字典序排列。 |
| `accumulate` | `<numeric>` | **求和** (可自定义初始值)。 |
| `iota` | `<numeric>` | **生成连续序列** (0, 1, 2, 3...)。 |
| `gcd` / `lcm` (C++17) | `<numeric>` | 最大公约数 / 最小公倍数。 |
| `inner_product` | `<numeric>` | 向量内积 (点积)。 |

---

这是 C++ `<algorithm>` 库中在做算法题（如 LeetCode、ACM）时**最高频**使用的函数总结。我按照**功能场景**进行了分类，并采用了与上一条相同的“**表格 + 代码**”格式。

所有代码默认包含：
```cpp
#include <iostream>
#include <vector>
#include <algorithm> // 核心头文件
#include <numeric>   // 部分数学函数在这里
using namespace std;
```

---

### 常用函数
#### 1. 排序 (Sorting)

做题第一步，往往是先把乱序变有序。

| 函数 | 说明 | 复杂度 |
| :--- | :--- | :--- |
| **`sort(beg, end)`** | **默认升序**。底层是快排混合。 | $O(N \log N)$ |
| **`sort(..., cmp)`** | 自定义排序规则（如降序、按结构体字段排）。 | $O(N \log N)$ |
| **`nth_element(...)`** | **Top K 神器**。将第 n 小的元素放到正确位置。 | **$O(N)$** |

```cpp
vector<int> v = {4, 1, 3, 2, 5};

// 1. 默认升序
sort(v.begin(), v.end()); // {1, 2, 3, 4, 5}

// 2. 降序 (使用 lambda)
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b; 
}); // {5, 4, 3, 2, 1}

// 3. 找第 2 小的元素 (下标为 1)
vector<int> nums = {10, 2, 5, 8, 1};
// 执行后，nums[1] 就是第2小的数，且左边都比它小
nth_element(nums.begin(), nums.begin() + 1, nums.end());
cout << nums[1] << endl; // 2
```

---

#### 2. 二分查找 (Binary Search)

**前提：** 序列必须**已排序**。返回的都是**迭代器**。

| 函数 | 说明 | 记忆口诀 |
| :--- | :--- | :--- |
| **`lower_bound`** | 找第一个 **$\ge$** val 的位置。 | 左边界 (常用) |
| **`upper_bound`** | 找第一个 **$>$** val 的位置。 | 右边界 |
| **`binary_search`** | 仅判断是否存在 (bool)。 | - |

```cpp
vector<int> v = {1, 2, 4, 4, 6};

// 1. 找第一个 >= 4 的位置
auto it = lower_bound(v.begin(), v.end(), 4);

if (it != v.end()) {
    // 转换为下标
    int index = it - v.begin(); 
    cout << "Index: " << index << endl; // 2
}

// 2. 统计 4 出现的次数 (利用 upper - lower)
auto last = upper_bound(v.begin(), v.end(), 4);
int count = last - it; // 4 - 2 = 2 个
```

---

#### 3. 最值查找 (Min / Max)

寻找最大最小值，或其位置。

| 函数 | 说明 | 注意点 |
| :--- | :--- | :--- |
| **`max(a, b)`** | 返回较大的值。也支持 `max({1,2,3})`。 | 返回值 |
| **`min(a, b)`** | 返回较小的值。 | 返回值 |
| **`max_element`** | 找最大元素的位置。 | **返回迭代器** (需 `*` 解引用) |
| **`min_element`** | 找最小元素的位置。 | **返回迭代器** |

```cpp
vector<int> v = {3, 1, 9, 7};

// 1. 直接比大小
int m = max(10, 20); // 20

// 2. 找数组中的最大值
auto it = max_element(v.begin(), v.end());
cout << "Max Value: " << *it << endl; // 9
cout << "Max Index: " << it - v.begin() << endl; // 2
```

---

#### 4. 排列与重组 (Permutation & Reorder)

处理全排列、反转、旋转。

| 函数 | 说明 | 典型应用 |
| :--- | :--- | :--- |
| **`reverse`** | 反转范围内的元素。 | 翻转字符串/数组 |
| **`next_permutation`** | 修改为下一个字典序排列。 | 全排列题目 |
| **`rotate`** | 循环移动数组。 | 数组右移 K 位 |

```cpp
vector<int> v = {1, 2, 3};

// 1. 反转
reverse(v.begin(), v.end()); // {3, 2, 1}

// 2. 全排列 (必须先排序)
sort(v.begin(), v.end()); // {1, 2, 3}
do {
    // 输出: 123, 132, 213 ...
    for(int n : v) cout << n; 
    cout << " ";
} while (next_permutation(v.begin(), v.end()));

// 3. 旋转 (把开头的 1 个移到最后)
// {1, 2, 3} -> {2, 3, 1}
rotate(v.begin(), v.begin() + 1, v.end());
```

---

#### 5. 去重与计数 (Count & Unique)

| 函数 | 说明 | 核心套路 |
| :--- | :--- | :--- |
| **`count`** | 统计某个值出现的次数。 | - |
| **`unique`** | **去重** (移除相邻重复元素)。 | 必须配合 `sort` 和 `erase` |

```cpp
vector<int> v = {1, 2, 2, 3, 3, 3};

// 1. 计数
int c = count(v.begin(), v.end(), 3); // 3

// 2. 真正的去重 (Standard Idiom)
// unique 将重复元素移到末尾，返回新逻辑结尾
auto new_end = unique(v.begin(), v.end());
// erase 真正删除垃圾数据
v.erase(new_end, v.end()); // v 变为 {1, 2, 3}
```

---

#### 6. 数学辅助 (Numeric) —— ⚠️ 位于 `<numeric>`

虽然不在 `<algorithm>`，但做数组题离不开它们。

| 函数 | 说明 |
| :--- | :--- |
| **`accumulate`** | **求和**。可指定初始值。 |
| **`iota`** | **生成连续序列** (0, 1, 2, ...)。 |

```cpp
#include <numeric> // 别忘了这个

vector<int> v = {1, 2, 3, 4};

// 1. 求和 (初始值设为 0)
// 注意：如果和可能超过 int，初始值要写 0LL
long long sum = accumulate(v.begin(), v.end(), 0LL); 

// 2. 填充 0 ~ N-1 (常用于并查集初始化)
vector<int> p(10);
iota(p.begin(), p.end(), 0); // {0, 1, 2, ..., 9}
```

---

