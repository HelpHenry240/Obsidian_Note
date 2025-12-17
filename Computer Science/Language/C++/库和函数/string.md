这是一个关于 C++ `<string>` 库常用函数的**实战总结**。为了方便记忆和查阅，我按照**功能场景**进行了分类，并为每个核心函数编写了简短的 `copy-paste` 级代码示例。

所有代码默认包含：
```cpp
#include <iostream>
#include <string>
using namespace std;
```

---

### 1. 基础信息查询 (Capacity & Check)

最基础的操作，用于了解字符串的状态。

| 函数 | 说明 |
| :--- | :--- |
| **`empty()`** | **推荐**。判断是否为空（比 `size() == 0` 快）。 |
| **`size()`** / **`length()`** | 返回字符数量。两者完全一样。 |

```cpp
string s = "Hello";

// 1. 判断空
if (!s.empty()) {
    cout << "String is not empty." << endl;
}

// 2. 获取长度
cout << "Length: " << s.size() << endl; // 输出 5
```

---

### 2. 元素访问 (Access)

读取特定位置的字符。

| 函数 | 说明 |
| :--- | :--- |
| **`s[i]`** | 下标访问。快，但不检查越界（越界程序可能崩溃）。 |
| **`at(i)`** | 安全访问。越界会抛出 `out_of_range` 异常。 |
| **`front()`** / **`back()`** | 直接获取首字符 / 尾字符。 |

```cpp
string s = "Code";

// 1. 下标访问
char c1 = s[1];      // 'o'

// 2. 首尾访问
char first = s.front(); // 'C'
char last  = s.back();  // 'e'

// 3. 修改字符
s[0] = 'M'; // 变成 "Mode"
```

---

### 3. 拼接与添加 (Append)

向字符串中增加内容。

| 函数 | 说明 |
| :--- | :--- |
| **`operator+=`** | **最常用**。可以拼接 string、char* 或 char。 |
| **`push_back(c)`** | 只能追加**一个字符**。 |
| **`append(str)`** | 类似 `+=`，但参数更灵活（如只追加 str 的一部分）。 |

```cpp
string s = "Hello";

// 1. 使用 += (推荐)
s += " World";  // "Hello World"
s += '!';       // "Hello World!"

// 2. 使用 push_back (仅限单字符)
s.push_back('?'); // "Hello World!?"

// 3. 使用 append (部分追加)
string other = "123456";
s.append(other, 0, 3); // 追加 other 从 0 开始的 3 个字符 -> "...?123"
```

---

### 4. 查找 (Find) —— ⚠️ 重点

核心常量：`string::npos`（表示“没找到”）。

| 函数 | 说明 |
| :--- | :--- |
| **`find(str)`** | 从左往右找，返回第一次出现的**下标**。 |
| **`rfind(str)`** | 从右往左找（找最后一次出现）。 |
| **`find_first_of(str)`** | 查找 str 中**任意一个字符**第一次出现的位置。 |

```cpp
string s = "apple.banana.jpg";

// 1. 查找子串
size_t pos = s.find("banana");
if (pos != string::npos) {
    cout << "Found at index: " << pos << endl; // 输出 6
} else {
    cout << "Not found" << endl;
}

// 2. 反向查找 (找文件后缀常用)
size_t dotPos = s.rfind('.'); // 找到最后一个 '.' 的位置 (12)
```

---

### 5. 子串提取 (Substring)

从大字符串中切出一块。

| 函数 | 说明 |
| :--- | :--- |
| **`substr(pos, len)`** | 从 `pos` 开始截取 `len` 个字符。**若省略 `len`，截取到末尾。** |

```cpp
string s = "2023-12-31";

// 1. 提取年份 (从0开始，取4个)
string year = s.substr(0, 4); // "2023"

// 2. 提取月份 (从5开始，取2个)
string month = s.substr(5, 2); // "12"

// 3. 提取剩余所有 (从8开始到结束)
string day = s.substr(8);      // "31"
```

---

### 6. 修改：插入、删除、替换 (Modify)

对字符串内部进行手术。

| 函数 | 说明 |
| :--- | :--- |
| **`insert(pos, str)`** | 在 `pos` 位置插入字符串。 |
| **`erase(pos, len)`** | 从 `pos` 开始删除 `len` 个字符。 |
| **`replace(pos, len, str)`** | 将指定范围的内容替换为新字符串。 |

```cpp
string s = "I love C++";

// 1. 插入
s.insert(2, "really "); // "I really love C++"

// 2. 替换 (把 "love" 换成 "hate")
// s.find("love") 返回位置，4 是 "love" 的长度
s.replace(s.find("love"), 4, "hate"); // "I really hate C++"

// 3. 删除 (删除 "really ")
s.erase(2, 7); // 变回 "I hate C++"
```

---

### 7. 类型转换 (Conversion) —— 现代 C++ 必备

数字与字符串互转。

| 函数 | 说明 |
| :--- | :--- |
| **`to_string(val)`** | 数值转字符串 (int, float, double...)。 |
| **`stoi(s)`** | String to Int。 |
| **`stod(s)`** | String to Double。 |
| **`c_str()`** | 获取 C 风格指针 (`const char*`)，用于兼容 C 接口。 |

```cpp
// 1. 数值 -> 字符串
int num = 100;
string s = "Value: " + to_string(num); // "Value: 100"

// 2. 字符串 -> 数值
string price = "19.99";
double d = stod(price); // 19.99
int i = stoi("45");     // 45

// 3. C++ String -> C API
string filename = "test.txt";
// fopen 需要 char*，不能直接传 string
FILE* fp = fopen(filename.c_str(), "r"); 
```

---

