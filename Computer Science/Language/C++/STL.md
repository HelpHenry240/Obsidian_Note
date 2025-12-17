STL（Standard Template Library）是 C++ 的**核心生产力工具**。如果说 C 语言是让程序员自己造轮子，C++ STL 就是提供了一整套世界顶级的轮子库。

---

### 第一部分：模板 (Templates) —— STL 的基石

**核心思想**：告诉编译器，“我不确定具体类型，请帮我生成代码”。
在 Python 中，类型是动态的，你写 `def add(a, b): return a + b`，`a` 和 `b` 可以是整数、浮点数甚至字符串。在 C++ 中，为了达到这种灵活性且保持高性能，我们使用模板。

#### 1. 函数模板 (Function Templates)
将类型参数化。使用 `template <typename T>`。

```cpp
#include <iostream>
using namespace std;

// 定义一个通用的交换函数
template <typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

int main() {
    int i1 = 10, i2 = 20;
    mySwap(i1, i2); // 编译器自动推导 T 为 int

    double d1 = 1.1, d2 = 2.2;
    mySwap(d1, d2); // 编译器自动推导 T 为 double
    
    return 0;
}
```

#### 2. 类模板 (Class Templates)
STL 的容器全是类模板（如 `vector<int>`, `vector<string>`）。

```cpp
template <typename T>
class Box {
public:
    T item;
    Box(T i) : item(i) {}
    T getItem() { return item; }
};

int main() {
    // 类模板不能自动推导，必须显式指定类型 <int>
    Box<int> b1(10); 
    Box<string> b2("Hello");
}
```

#### 3. 模板特化 (Specialization) —— ⚠️ 进阶易错点
有时候，通用的逻辑对某些特定类型（比如 `bool` 或 `char*`）不适用，我们需要“特殊对待”。

*   **全特化**：对所有模板参数都进行特定定义。
```cpp
// 通用版本
template <typename T>
class Calculator { ... };

// 针对 bool 类型的特化版本（因为 bool 运算逻辑可能不同）
template <>
class Calculator<bool> {
    // 专门为 bool 写的逻辑
};
```

---

### 第二部分：STL 六大组件 —— C++ 的军火库

STL 将数据结构（容器）和算法分离开了，通过迭代器连接。

#### 1. 容器 (Containers) —— 装数据的盒子

| 分类                   | 容器名                                                  | 底层实现     | 特点                                          | Python 对应           | 适用场景                |
| :------------------- | :--------------------------------------------------- | :------- | :------------------------------------------ | :------------------ | :------------------ |
| **序列式**              | **`vector`**[vector](vector.md)                      | **动态数组** | **首选**。尾部插入快 `O(1)`，随机访问快 `O(1)`，中间插入慢。     | `list` (最像)         | **90%的情况都用它**。      |
|                      | `list`[list](list.md)                                | 双向链表     | 任意位置插入删除快 `O(1)`，不支持随机访问。                   | `collections.deque` | 需要频繁在中间插入删除时。       |
|                      | `deque`                                              | 双端队列     | 头尾插入都快。                                     | `deque`             | 需要在头部插入元素时。         |
| **关联式**              | **`map`**                                            | **红黑树**  | Key-Value 对，**自动排序** (按 Key)。查找 `O(log n)`。 | -                   | 需要有序遍历 Key 时。       |
|                      | `set`                                                | 红黑树      | 只有 Key，自动去重且排序。                             | `set` (但不排序)        | 需要去重且有序时。           |
| **无序式** <br> (C++11) | **`unordered_map`**[unordered_map](unordered_map.md) | **哈希表**  | Key-Value 对，无序。查找平均 `O(1)`。                 | **`dict`** (本质)     | **追求极致查找速度**，不在乎顺序。 |
|                      | `unordered_set`[unordered_set](unordered_set.md)     | 哈希表      | 只有 Key，去重，无序。                               | `set`               | 只需要快速去重。            |

**代码演示 (vector 与 map):**
```cpp
#include <vector>
#include <map>
#include <string>
#include <iostream>

using namespace std;

int main() {
    // 1. Vector (动态数组)
    vector<int> v = {1, 2, 3};
    v.push_back(4); // 尾部追加
    cout << v[0] << endl; // 随机访问
    
    // 2. Map (有序字典)
    map<string, int> scores;
    scores["Tom"] = 90;
    scores["Jerry"] = 80;
    // Map 会自动按 Key (名字) 排序
}
```

#### 2. 迭代器 (Iterators) —— 盲人摸象的手
迭代器是一种**广义的指针**。它让你不需要知道底层是数组还是链表，用统一的方式遍历容器。

*   `begin()`: 指向第一个元素。
*   `end()`: 指向**最后一个元素的下一个位置**（左闭右开区间 `[begin, end)`）。

**遍历容器的进化史：**

```cpp
vector<int> v = {10, 20, 30};

// 方式 1: 传统 for 循环 (仅限 vector/deque，因为支持下标)
for(int i=0; i<v.size(); ++i) { ... }

// 方式 2: 迭代器 (适用于所有容器，如 list, map)
// iterator 本质像指针，用 * 解引用
for(vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
    cout << *it << " "; 
}

// 方式 3: C++11 范围 for 循环 (The Python Way) —— 推荐 ✅
for(int val : v) {
    cout << val << " ";
}
```

#### 3. 算法 (Algorithms)
[algorithm](algorithm.md)
位于 `<algorithm>` 头文件。STL 提供了几百个通用算法，直接作用于迭代器范围。

*   `std::sort(v.begin(), v.end())`: 排序 (快排/堆排混合)。
*   `std::find(v.begin(), v.end(), 20)`: 查找。
*   `std::reverse(...)`: 反转。

#### 4. 适配器 (Adapters)
它们不是真正的容器，而是把底层容器（如 deque）包装了一下，限制了接口。
*   `stack`: 先进后出 (LIFO)。[stack](stack.md)
*   `queue`: 先进先出 (FIFO)。[queue](queue.md)
*   `priority_queue`: 优先队列（最大堆），每次弹出的都是优先级最高（最大）的元素。

---

### 第三部分：Lambda 表达式 (C++11) —— 现代 C++ 的灵魂

在 C++11 之前，如果你想让 `sort` 按照从大到小排序，你得专门写一个比较函数或者仿函数类，非常啰嗦。Lambda 允许你定义**匿名函数**。

**语法结构：** `[捕获列表](参数列表) { 函数体 }`

#### 3.1 基础用法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> v = {1, 5, 3, 9};

    // 默认 sort 是从小到大
    // 使用 Lambda 自定义排序规则：从大到小
    sort(v.begin(), v.end(), [](int a, int b) {
        return a > b; // 类似于 Python 的 lambda a, b: a > b
    });
}
```

#### 3.2 捕获列表 `[]` (Capture List) —— ⚠️ 重点
Lambda 最大的魔力在于它可以“捕获”外部作用域的变量。

*   `[]`: 不捕获任何外部变量。
*   `[=]`: 按**值**捕获外部所有变量（只读拷贝）。
*   `[&]`: 按**引用**捕获外部所有变量（可以修改外部变量）。
*   `[x]`: 只按值捕获变量 x。

**代码示例：**
```cpp
int multiple = 2;
vector<int> v = {1, 2, 3};

// for_each 遍历算法
// 我们想把 v 里的每个数都打印出来，并且打印出它乘以 multiple 的结果
for_each(v.begin(), v.end(), [multiple](int val) {
    // multiple 是从外部捕获进来的
    cout << val * multiple << endl; 
});
```

---

### ⚡ 总结与对比速查表

| 概念 | Python | C++ STL | 备注 |
| :--- | :--- | :--- | :--- |
| **通用编程** | 动态类型自然支持 | **模板 (Template)** | 编译期生成代码，高性能 |
| **列表** | `list` (混合体) | **`vector`** (数组), `list` (链表) | 默认用 `vector` |
| **字典** | `dict` (哈希) | `map` (有序树), **`unordered_map`** (哈希) | 需要顺序用 `map`，求快用 `unordered_map` |
| **遍历** | `for x in list` | `for(auto x : vec)` (C++11) | 语法非常像了 |
| **匿名函数** | `lambda x: x+1` | `[](int x){ return x+1; }` | C++ Lambda 主要是为了配合 STL 算法 |

### 🚀 学习路径建议
1.  **先用起来**：不要去背 `vector` 的所有函数，先会用 `push_back`, `[]`, 迭代器遍历。
2.  **分清 Map**：面试必问 `map` (红黑树, O(logN), 有序) 和 `unordered_map` (哈希表, O(1), 无序) 的区别。
3.  **拥抱 Lambda**：只要用到 `sort`, `find_if` 等算法，立刻想到用 Lambda，代码会干净很多。