## 1. **函数想改外部变量，必用引用 (`&`) 或指针 (`*`)**。这是 C++ 最基础也最重要的规则之一。默认的参数传递都是“复印件”。
## 2. **绝不访问未初始化的 `vector` 或数组的索引**。在使用 `[]` 访问 `vector` 之前，必须确保该位置已经通过 `push_back` 或预设大小等方式分配了空间。
## 3. 循环输入判断
主要问题：

1. **类型不匹配**：`cin >> data` 返回的是 `istream&`（输入流对象），而 `endl` 是 `ostream` 操纵符，两者不能直接比较。

2. **逻辑错误**：即使能比较，这个条件判断也不正确，应该检查读取是否成功。

### 正确写法：

#### 方法1：检查读取是否成功
```cpp
while(cin >> data) {
    preorder[index++] = data;
}
```

#### 方法2：读取到特定值结束
```cpp
while(cin >> data && data != endMarker) {  // endMarker是某个结束标志值
    preorder[index++] = data;
}
```

#### 方法3：读取一行数据
```cpp
string line;
while(getline(cin, line) && !line.empty()) {
    istringstream iss(line);
    while(iss >> data) {
        preorder[index++] = data;
    }
}
```

#### 方法4：读取固定数量
```cpp
for(int i = 0; i < n && cin >> data; i++) {
    preorder[index++] = data;
}
```

#### 如果是要读取到换行符结束：

```cpp
int data;
while(cin >> data) {
    preorder[index++] = data;
    
    // 检查下一个字符是否是换行符
    if(cin.peek() == '\n') {
        cin.get(); // 消耗换行符
        break;
    }
}
```

最常用的模式就是 `while(cin >> data)`，它会自动在遇到文件结束、类型不匹配或错误时停止。