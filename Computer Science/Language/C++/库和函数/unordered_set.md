`unordered_set` 是 C++ 标准库中的一种哈希集合实现，用于存储不重复的元素，常见使用场景是对元素进行去重。

本站后面的章节会介绍哈希集合的实现原理和代码实现，这里只介绍一下它的常用 API。

初始化方法：

```cpp
#include <unordered_set>
using namespace std;

// 初始化一个空的哈希集合 set
unordered_set<int> uset;

// 初始化一个包含一些元素的哈希集合 set
unordered_set<int> uset{1, 2, 3, 4};
```

`unordered_set` 的常用方法：

```cpp
#include <iostream>
#include <unordered_set>
using namespace std;

int main() {
    // 初始化哈希集合
    unordered_set<int> hashset{1, 2, 3, 4};

    // 检查哈希集合是否为空，输出：0 (false)
    cout << hashset.empty() << endl;

    // 获取哈希集合的大小，输出：4
    cout << hashset.size() << endl;

    // 查找指定元素是否存在
    // 输出：Element 3 found.
    if (hashset.contains(3)) {
        cout << "Element 3 found." << endl;
    } else {
        cout << "Element 3 not found." << endl;
    }

    // 插入一个新的元素
    hashset.insert(5);

    // 删除一个元素
    hashset.erase(2);
    // 输出：Element 2 not found.
    if (hashset.contains(2)) {
        cout << "Element 2 found." << endl;
    } else {
        cout << "Element 2 not found." << endl;
    }

    // 遍历哈希集合
    // 输出（顺序可能不同）：
    // 1
    // 3
    // 4
    // 5
    for (const auto &element : hashset) {
        cout << element << endl;
    }

    return 0;
}
```