`unordered_map` 是 C++ 标准库中的一种哈希表实现，它提供了基于键值对（key-value）的存储，提供了常数时间复杂度的查找、插入和删除键值对的操作。

初始化方法：

```cpp
#include <unordered_map>
using namespace std;

// 初始化一个空的哈希表 map
unordered_map<int, string> hashmap;

// 初始化一个包含一些键值对的哈希表 map
unordered_map<int, string> hashmap{{1, "one"}, {2, "two"}, {3, "three"}};
```

`unordered_map` 的常用方法如下。

特别注意：访问不存在的键会自动插入键值对

**在 C++ 的哈希表中，如果你访问一个不存在的键，它会自动创建这个键，对应的值是默认构造的值**。

这一点和其他语言不同，需要格外注意。记住访问值之前要先判断键是否存在，否则可能会意外地创建新键，导致算法出错。详见下面的示例。

```cpp
#include <iostream>
#include <unordered_map>
using namespace std;

int main() {
    // 初始化哈希表
    unordered_map<int, string> hashmap{{1, "one"}, {2, "two"}, {3, "three"}};

    // 检查哈希表是否为空，输出：0 (false)
    cout << hashmap.empty() << endl;

    // 获取哈希表的大小，输出：3
    cout << hashmap.size() << endl;

    // 查找指定键是否存在
    // 注意 contains 方法是 C++20 新增的
    // 输出：Key 2 -> two
    if (hashmap.contains(2)) {
        cout << "Key 2 -> " << hashmap[2] << endl;
    } else {
        cout << "Key 2 not found." << endl;
    }

    // 获取指定键对应的值，若不存在会返回默认构造的值
    // 输出空字符串
    cout << hashmap[4] << endl;

    // 插入一个新的键值对
    hashmap[4] = "four";

    // 获取新插入的值，输出：four
    cout << hashmap[4] << endl;

    // 删除键值对
    hashmap.erase(3);

    // 检查删除后键 3 是否存在
    // 输出：Key 3 not found.
    if (hashmap.contains(3)) {
        cout << "Key 3 -> " << hashmap[3] << endl;
    } else {
        cout << "Key 3 not found." << endl;
    }

    // 遍历哈希表
    // 输出（顺序可能不同）：
    // 4 -> four
    // 2 -> two
    // 1 -> one
    for (const auto &pair: hashmap) {
        cout << pair.first << " -> " << pair.second << endl;
    }

    // 特别注意，访问不存在的键会自动创建这个键
    unordered_map<int, string> hashmap2;

    // 键值对的数量是 0
    cout << hashmap2.size() << endl; // 0

    // 访问不存在的键，会自动创建这个键，对应的值是默认构造的值
    cout << hashmap2[1] << endl; // empty string
    cout << hashmap2[2] << endl; // empty string

    // 现在键值对的数量是 2
    cout << hashmap2.size() << endl; // 2

    return 0;
}
```