`queue` 是 C++ 标准库中的队列容器，基于先进先出（FIFO）的原则。队列适用于只允许从一端（队尾）添加元素、从另一端（队头）移除元素的场景。

关于队列的实现原理和使用场景，本站后面的章节会介绍，这里只介绍一下它的常用 API。

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    // 初始化一个空的整型队列 q
    queue<int> q;

    // 在队尾添加元素
    q.push(10);
    q.push(20);
    q.push(30);

    // 检查队列是否为空，输出：false
    cout << q.empty() << endl;

    // 获取队列的大小，输出：3
    cout << q.size() << endl;

    // 获取队列的队头和队尾元素，输出：10 和 30
    cout << q.front() << " " << q.back() << endl;

    // 删除队头元素
    q.pop();

    // 输出新的队头元素：20
    cout << q.front() << endl;

    return 0;
}
```