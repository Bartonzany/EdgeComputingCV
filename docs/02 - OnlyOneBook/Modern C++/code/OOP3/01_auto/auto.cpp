#include <iostream>
#include <vector>

// 使用 auto 作为函数返回类型（C++14 起可用 trailing-return-type）
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

int main() {
    // 基础类型推导
    auto x   = 10;         // x -> int
    auto y   = 3.14;       // y -> double
    auto str = "Hello";    // str -> const char*

    std::cout << "x = " << x << ", y = " << y << ", str = " << str << "\n\n";

    // 指针与引用推导
    int var = 0;

    auto*       a = &var;    // a -> int*, auto 推导为 int
    auto        b = &var;    // b -> int*, auto 推导为 int*
    auto &      c = var;     // c -> int&, auto 推导为 int
    auto        d = c;       // d -> int, auto 推导为 int
    const auto  e = var;     // e -> const int, auto 推导为 int
    auto        f = e;       // f -> int, 忽略顶层 const
    const auto &g = var;     // g -> const int&, auto 推导为 int
    auto &      h = g;       // h -> const int&, auto 推导为 const int

    std::cout << "a = " << a << ", *a = " << *a << "\n";
    std::cout << "b = " << b << ", *b = " << *b << "\n";
    std::cout << "c = " << c << ", d = " << d << "\n";
    std::cout << "e = " << e << ", f = " << f << "\n";
    std::cout << "g = " << g << ", h = " << h << "\n\n";

    // 容器迭代器与范围 for
    std::vector<int> vec = { 1, 2, 3 };
    auto             it  = vec.begin();    // 推导为 std::vector<int>::iterator
    std::cout << "First element: " << *it << std::endl;

    std::cout << "vec elements: ";
    for (auto val: vec) {    // val 推导为 int
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    // Lambda 表达式
    auto lambda = [](int x) {
        return x * 2;
    };
    std::cout << "lambda(5) = " << lambda(5) << "\n\n";

    // 表达式类型推导
    int    i = 5;
    double j = 3.14;
    auto   k = i + j;    // 推导为 double，避免截断
    std::cout << "k = i + j = " << k << " (" << typeid(k).name() << ")\n\n";

    // 函数模板返回值推导
    auto result = add(3, 4.5);    // 推导为 double
    std::cout << "add(3, 4.5) = " << result << " (" << typeid(result).name() << ")\n";

    return 0;
}