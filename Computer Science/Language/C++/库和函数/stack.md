栈是一种后进先出（LIFO）的数据结构，栈适用于只允许在一端（栈顶）添加或移除元素的场景。

`stack` 是 C++ 标准库中的栈容器，关于栈的实现原理和使用场景，本站后面的章节会介绍，这里只介绍一下它的常用 API。

```cpp
#include <iostream>
#include <stack>
using namespace std;

int main() {

    // 初始化一个空的整型栈 s
    stack<int> s;

    // 向栈顶添加元素
    s.push(10);
    s.push(20);
    s.push(30);

    // 检查栈是否为空，输出：false
    cout << s.empty() << endl;

    // 获取栈的大小，输出：3
    cout << s.size() << endl;

    // 获取栈顶元素，输出：30
    cout << s.top() << endl;

    // 删除栈顶元素
    s.pop();

    // 输出新的栈顶元素：20
    cout << s.top() << endl;

    return 0;
}
```