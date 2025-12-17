`vector` 的初始化方法如下：

```cpp
#include <vector>

int n = 7, m = 8;

// 初始化一个 int 型的空数组 nums
vector<int> nums;

// 初始化一个大小为 n 的数组 nums，数组中的值默认都为 0
vector<int> nums(n);

// 初始化一个元素为 1, 3, 5 的数组 nums
vector<int> nums{1, 3, 5};

// 初始化一个大小为 n 的数组 nums，其值全都为 2
vector<int> nums(n, 2);

// 初始化一个二维 int 数组 dp
vector<vector<int>> dp;

// 初始化一个大小为 m * n 的布尔数组 dp，
// 其中的值都初始化为 true
vector<vector<bool>> dp(m, vector<bool>(n, true));
```

`vector` 的常用操作：

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n = 10;
    // 数组大小为 10，元素值都为 0
    vector<int> nums(n);
    // 输出 0 (false)
    cout << nums.empty() << endl;
    // 输出：10
    cout << nums.size() << endl;

    // 在数组尾部插入一个元素 20
    nums.push_back(20);
    // 输出：11
    cout << nums.size() << endl;

    // 得到数组最后一个元素的引用
    // 输出：20
    cout << nums.back() << endl;

    // 删除数组的最后一个元素（无返回值）
    nums.pop_back();
    // 输出：10
    cout << nums.size() << endl;

    // 可以通过方括号直接取值或修改
    nums[0] = 11;
    // 输出：11
    cout << nums[0] << endl;

    // 在索引 3 处插入一个元素 99
    nums.insert(nums.begin() + 3, 99);

    // 删除索引 2 处的元素
    nums.erase(nums.begin() + 2);

    // 交换 nums[0] 和 nums[1]
    swap(nums[0], nums[1]);

    // 遍历数组
    // 0 11 99 0 0 0 0 0 0 0
    for (int i = 0; i < nums.size(); i++) {
        cout << nums[i] << " ";
    }
    cout << endl;
}
```

