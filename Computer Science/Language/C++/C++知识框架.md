构建C++的知识框架是一个系统性的工程，因为C++是一门“多范式”且深度极深的语言。这个框架分为 **“基础 -> 核心 -> 进阶 -> 现代特性 -> 工程化”** 五个层级。

---

### 第一阶段：坚实基础 (The C in C++)
[Better C](Better%20C.md)
这是C++的根基，主要涵盖类似C语言的语法，但要用C++的思维去理解。

1.  **基础语法**
    *   **数据类型**：整型、浮点型、布尔型、字符型、`auto` (自动推导)。
    *   **控制结构**：`if/else`, `switch`, `for` (包括基于范围的for循环), `while`, `do-while`。
    *   **函数**：函数声明与定义、参数传递（值传递 vs **引用传递** vs 指针传递）、函数重载、内联函数 (`inline`)。
    *   **预处理**：`#include`, `#define` (宏), `#ifdef` (条件编译)。

2.  **复合类型**
    *   **数组与字符串**：原生数组 vs `std::vector`，C风格字符串 (`char*`) vs **`std::string`** (强烈建议尽早习惯用 `std::string`)。
    *   **指针与引用**：
        *   指针的本质、空指针 (`nullptr` vs `NULL`)。
        *   **引用 (`&`)**：C++区别于C的重要特性，必须掌握。
        *   `const` 限定符：常量指针 vs 指针常量。

---

### 第二阶段：面向对象 (Object-Oriented Programming)
[面向对象(Class)](面向对象(Class).md)
这是C++经典的编程范式，核心在于“封装、继承、多态”。

1.  **类与对象 (Class & Object)**
    *   **封装**：`public`, `private`, `protected` 访问权限。
    *   **构造与析构**：构造函数、析构函数、初始化列表。
    *   **特殊成员函数**：拷贝构造、拷贝赋值。
    *   **静态成员**：`static` 成员变量与函数。
    *   **This指针**。

2.  **继承与多态 (Inheritance & Polymorphism)**
    *   **继承方式**：公有、保护、私有继承。
    *   **虚函数**：`virtual` 关键字，虚函数表 (vtable) 原理。
    *   **重写 (Override)**：`override`, `final` 关键字 (C++11)。
    *   **纯虚函数与抽象类**：接口的定义。
    *   **多重继承**与**虚继承** (解决菱形继承问题)。

---

### 第三阶段：内存管理与 RAII (The Soul of C++)
这是C++最难也最精华的部分，区别于Java/Python等带GC的语言。

1.  **内存模型**
    *   栈 (Stack) vs 堆 (Heap) vs 全局/静态区。
    *   `new` / `delete` 与 `malloc` / `free` 的区别。

2.  **RAII (资源获取即初始化)**
    *   **核心思想**：利用栈对象的生命周期管理堆资源（这是C++防止内存泄漏的法宝）。
    *   **Rule of Three / Five / Zero**（三/五/零法则）：何时需要手写拷贝/移动构造函数。

3.  **智能指针 (Smart Pointers) - C++11**
    *   **`std::unique_ptr`**：独占所有权（最常用）。
    *   **`std::shared_ptr`**：共享所有权（引用计数）。
    *   **`std::weak_ptr`**：解决 shared_ptr 的循环引用问题。

4.  **移动语义 (Move Semantics) - C++11**
    *   左值 (lvalue) vs 右值 (rvalue)。
    *   **右值引用 (`&&`)**。
    *   **`std::move`** 和 **`std::forward`** (完美转发)。
    *   移动构造函数与移动赋值操作符（大幅提升性能的关键）。

---

### 第四阶段：泛型编程与 STL (Standard Template Library)
[STL](STL.md)
不写模板可能还是C++程序员，但不会用STL绝对不是合格的C++程序员。
1.  **模板 (Templates)**
    *   函数模板。
    *   类模板。
    *   模板特化 (全特化/偏特化)。

2.  **STL 六大组件**
    *   **容器 (Containers)**：
        *   序列式：`vector` (动态数组), `list` (双向链表), `deque` (双端队列).
        *   关联式：`map`, `set`, `multimap` (基于红黑树).
        *   无序式 (C++11)：`unordered_map`, `unordered_set` (基于哈希表).
    *   **迭代器 (Iterators)**：连接容器与算法的桥梁。
    *   **算法 (Algorithms)**：`std::sort`, `std::find`, `std::transform`, `std::for_each` 等。
    *   **仿函数 (Functors)** / 函数对象。
    *   **适配器 (Adapters)**：`stack`, `queue`, `priority_queue`。
    *   **分配器 (Allocators)**：(通常了解即可，高阶内容)。

3.  **Lambda 表达式**
    *   匿名函数，捕获列表 `[]`，用于配合 STL 算法极其方便。

---

### 第五阶段：现代 C++ 特性 (Modern C++ Evolution)
C++11 是分水岭，C++14/17/20 持续演进。

1.  **C++11/14 (现代化基石)**
    *   `auto` 与 `decltype`。
    *   范围 for 循环 (`for(auto& x : vec)`).
    *   `nullptr`.
    *   `constexpr` (编译期常量).
    *   `std::function` 与 `std::bind`.

2.  **C++17 (语法糖与标准库增强)**
    *   结构化绑定 (`auto [x, y] = pair`).
    *   `if constexpr`.
    *   `std::optional`, `std::variant`, `std::any` (现代类型安全的联合体/空值处理).
    *   `std::filesystem`.

3.  **C++20 (重大变革)**
    *   **Modules (模块)**：告别头文件，加速编译。
    *   **Concepts (概念)**：约束模板类型，改善错误信息。
    *   **Coroutines (协程)**：原生支持异步编程。
    *   **Ranges (范围库)**：函数式的管道操作处理容器。

---

### 第六阶段：高阶与底层 (Advanced & Systems)
这部分决定了你是否能编写高性能库或系统级软件。

1.  **并发编程 (Concurrency)**
    *   `std::thread`.
    *   互斥量 `std::mutex`, 锁 `std::unique_lock`, `std::lock_guard`.
    *   条件变量 `std::condition_variable`.
    *   原子操作 `std::atomic` (无锁编程基础).
    *   异步任务 `std::future`, `std::async`.

2.  **元编程 (Metaprogramming)**
    *   模板元编程 (TMP)：在编译期进行计算。
    *   Type Traits (`std::is_same`, `std::enable_if`).

3.  **异常处理与类型转换**
    *   `try-catch-throw`.
    *   四种 Cast：`static_cast`, `dynamic_cast`, `const_cast`, `reinterpret_cast`.

---

### 第七阶段：工程化与工具 (Tooling)
写代码不仅是写语法，还得会构建和调试。

1.  **构建系统**
    *   **CMake** (事实上的行业标准，必须掌握)。
    *   Makefile (了解原理)。

2.  **调试与性能分析**
    *   **GDB / LLDB**：命令行调试。
    *   **Valgrind / AddressSanitizer**：内存泄漏与越界检测。
    *   **Perf / GProf**：性能热点分析。

3.  **代码规范**
    *   Google C++ Style Guide 或 LLVM Style Guide。
    *   Clang-Format, Clang-Tidy。

---

### 学习路径

1.  **入门**：掌握 **第一阶段** 和 **第二阶段**，加上 `std::string` 和 `std::vector`。
    *   *书籍推荐*：《C++ Primer》(虽然厚，但是圣经)。
2.  **进阶**：死磕 **第三阶段 (内存/RAII)** 和 **第四阶段 (STL)**。这是C++最好用也最容易出错的地方。
    *   *书籍推荐*：《Effective C++》(必读，讲的是最佳实践)。
3.  **提升**：学习 **第五阶段 (Modern C++)** 和 **第六阶段 (并发)**。
    *   *书籍推荐*：《Effective Modern C++》。
4.  **实战**：结合 **第七阶段**，自己写一个小项目（如简单的Web服务器、JSON解析器或光线追踪渲染器）。

这个框架涵盖了从“会写代码”到“写出高性能、安全代码”的所有知识点。