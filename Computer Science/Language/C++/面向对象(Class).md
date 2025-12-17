
我们分为四个阶段：**基础框架** -> **生命周期** -> **继承与多态** -> **核心原理**。

---

### 第一阶段：类的定义与封装 (Class & Encapsulation)

C++ 是**静态类型**语言，Python 是**动态类型**语言。这导致了两者在定义类时最大的不同：C++ 必须明确声明成员变量的类型，且权限控制极其严格。

#### 1.1 基本语法对比

**场景**：定义一个“学生”类，有名字和年龄。

**Python 写法：**
```python
class Student:
    def __init__(self, name, age):
        self.name = name  # 成员变量随时可以在函数里添加
        self.age = age

    def study(self):
        print(f"{self.name} is studying.")

# 使用
s = Student("Tom", 18)
s.study()
```

**C++ 写法：**
```cpp
#include <iostream>
#include <string>
using namespace std;

// ⚠️ 易错点1：C++ 类定义结尾必须有分号 ';'
class Student {
public: // 1. 访问权限说明符
    // 2. 成员变量必须提前声明类型
    string name; 
    int age;

    // 3. 构造函数 (Constructor) 对应 Python 的 __init__
    Student(string n, int a) {
        name = n;
        age = a;
    }

    // 4. 成员函数
    // C++ 中不需要显式写 self 参数，内部自动隐含了 this 指针
    void study() {
        cout << name << " is studying." << endl;
    }
};

int main() {
    // 实例化对象
    Student s("Tom", 18); 
    s.study(); // 使用点号 . 访问
    return 0;
}
```

#### 1.2 访问权限 (Access Control) —— ⚠️ 重点区别

*   **Python**：主要靠约定。`_name` 表示“请把它当私有”，但你非要访问也能访问。
*   **C++**：**编译器强制执行**。如果你试图访问 `private` 成员，代码直接编译报错，跑都跑不起来。

| 关键字 | C++ 含义 | Python 对应 |
| :--- | :--- | :--- |
| **`public`** | 任何人都可以访问 | 默认情况 (`self.name`) |
| **`private`** | 只有**本类内部**可以访问 | 约定 `__name` (双下划线) |
| **`protected`** | 本类和**子类**可以访问 | 约定 `_name` (单下划线) |

**代码示例：**
```cpp
class Box {
public:
    int width;   // 谁都能改
private:
    int length;  // 只有 Box 内部函数能改
};

int main() {
    Box b;
    b.width = 10;  // ✅ OK
    // b.length = 20; // ❌ 编译报错！无法访问 private 成员
}
```

---

### 第二阶段：对象的生命周期 (Lifecycle)

这是 C++ 与 Python 最本质的区别。Python 有垃圾回收（GC），你只管创建，不管销毁。**C++ 必须由程序员掌控生死**，或者利用栈（Stack）的自动特性。

#### 2.1 构造函数与析构函数

*   **构造函数 (`Constructor`)**：对象出生时调用。
    *   Python: `__init__`
    *   C++: `ClassName()`
*   **析构函数 (`Destructor`)**：对象死亡前一刻调用。
    *   Python: `__del__` (很少手动写)
    *   C++: `~ClassName()` (经常写，用于释放内存)

#### 2.2 实例化对象的两种方式 —— ⚠️ 高危易错点

在 Python 中，所有对象都在堆（Heap）上，变量只是引用。但在 C++ 中，你有两个选择：

**方式一：在栈上创建 (Stack Allocation) —— 推荐，C++ 特色**
```cpp
void func() {
    Student s("Tom", 18); // 像定义 int a = 10 一样
    s.study();
} // ✅ 函数结束，s 自动销毁，析构函数自动调用
```

**方式二：在堆上创建 (Heap Allocation) —— 像 Python**
```cpp
void func() {
    // 使用 new 关键字，返回的是指针
    Student* p = new Student("Tom", 18); 
    
    // ⚠️ 易错点：指针调用成员用箭头 '->' 而不是 '.'
    p->study(); 

    // ❌ 必须手动 delete，否则内存泄漏！Python 不需要这步
    delete p; 
}
```

**对比总结：**
*   **Python**: `s = Student()` (全是引用，自动回收)
*   **C++ (栈)**: `Student s;` (自动回收，速度快，用 `.`)
*   **C++ (堆)**: `Student* s = new Student();` (手动回收，灵活，用 `->`)

---

### 第三阶段：继承 (Inheritance)

语法非常相似，但 C++ 多了一个继承权限的概念。

**Python:**
```python
class Dog(Animal):
    pass
```

**C++:**
```cpp
// ⚠️ 易错点：通常使用 public 继承，保留父类的权限
class Dog : public Animal {
public:
    void bark() { cout << "Woof!" << endl; }
};
```
*   如果不写 `public`，默认是 `private` 继承（父类的 public 成员到了子类变成 private），这通常不是我们想要的“是一个(is-a)”的关系。所以**99% 的情况请加 `public`**。

---

### 第四阶段：多态 (Polymorphism) —— ⚠️ 核心难点

这是 C++ 初学者最容易掉坑的地方。

*   **Python**：**天生多态**。只要对象有那个方法，就能调用（鸭子类型）。
*   **C++**：**默认静态绑定**。如果不特殊说明，编译器在编译时就决定了调用哪个函数，不管运行时对象到底是谁。

必须要用 **`virtual` (虚函数)** 关键字来开启多态。

#### 4.1 没有 `virtual` 的悲剧

```cpp
class Animal {
public:
    void speak() { cout << "Animal moves" << endl; }
};

class Cat : public Animal {
public:
    void speak() { cout << "Meow" << endl; }
};

void makeItSpeak(Animal* a) {
    a->speak(); // ⚠️ 问题来了！
}

int main() {
    Cat* c = new Cat();
    makeItSpeak(c); // 输出 "Animal moves" ❌
    // 为什么？因为 C++ 看到参数类型是 Animal*，默认直接绑定 Animal::speak()
}
```

#### 4.2 加上 `virtual` (虚函数)

```cpp
class Animal {
public:
    // ✅ 加上 virtual，告诉编译器：运行时再看具体是谁
    virtual void speak() { cout << "Animal moves" << endl; }
};

class Cat : public Animal {
public:
    // C++11 建议加上 override，明确表示我在覆盖父类方法
    void speak() override { cout << "Meow" << endl; }
};

// ... 现在 makeItSpeak(c) 就会输出 "Meow" 了 ✅
```

#### 4.3 纯虚函数与抽象类 (Abstract Class)

Python 使用 `ABC` 模块定义抽象基类。C++ 使用**纯虚函数**。

```cpp
class Shape {
public:
    // "= 0" 表示这是纯虚函数，没有实现
    // 包含纯虚函数的类就是抽象类，不能实例化
    virtual void draw() = 0; 
};

class Circle : public Shape {
public:
    void draw() override { cout << "Drawing Circle"; }
};
```

---

### ⚡ 深度总结与易错点标注

| 特性 | Python | C++ | ⚠️ 易错/注意点 |
| :--- | :--- | :--- | :--- |
| **self / this** | 函数定义需显式写 `self` | 隐式包含 `this` 指针 | 在 C++ 类内部访问成员直接写名字，不用 `this->name` (除非重名)。 |
| **初始化** | `__init__` | 构造函数 `Class()` | C++ 还有**初始化列表**（`Class(): a(1){}`），效率更高。 |
| **对象创建** | `s = Class()` (引用) | `Class s;` (栈值) <br> `Class* s = new Class();` (堆指针) | 分清 `.` (对象) 和 `->` (指针)。 |
| **多态** | 默认支持 | **必须加 `virtual`** | 父类析构函数**必须**也是 `virtual`，否则删除父类指针时，子类析构不会执行（内存泄漏）。 |
| **分号** | 不需要 | 类定义结束必须有 `;` | 漏掉分号报错信息非常诡异。 |
| **接口/抽象** | `ABC` 模块 | 纯虚函数 `= 0` | 只要有一个纯虚函数，类就无法实例化。 |

#### 学习建议
从 Python 转 C++，最需要扭转的思维是：
1.  **内存意识**：对象到底在栈上还是堆上？谁负责清理它？
2.  **类型意识**：多态不是天生的，需要显式声明 `virtual` 才能获得“动态”的特性。