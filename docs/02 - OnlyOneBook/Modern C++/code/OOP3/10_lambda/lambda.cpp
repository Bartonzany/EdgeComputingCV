#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // 示例 1：简单的 Lambda 表达式
    [] {
        std::cout << "Hello, Lambda!" << std::endl;
    }();

    auto l = [] {
        std::cout << "Hello, Lambda!" << std::endl;
    };
    l();

    // 示例 2：捕获外部变量（按值）
    int  id = 0;
    auto f  = [id]() mutable {
        std::cout << "ID: " << id << std::endl;
        ++id;    // 修改按值捕获的副本，不影响外部变量
    };
    f();                                                // 输出 ID: 0
    std::cout << "External ID: " << id << std::endl;    // 输出 External ID: 0

    // 示例 3：捕获引用
    id     = 42;
    auto g = [&id]() {
        std::cout << "ID: " << id << std::endl;
        ++id;    // 修改外部变量
    };
    g();                                                // 输出 ID: 42
    std::cout << "External ID: " << id << std::endl;    // 输出 External ID: 43

    // 示例 4：结合 STL 算法
    std::vector<int> vi = { 5, 28, 50, 83, 70, 590, 245, 59, 24 };
    int              x  = 30;
    int              y  = 100;

    // 使用 Lambda 表达式过滤范围内的元素 (30, 100)
    vi.erase(std::remove_if(vi.begin(), vi.end(),
                            [x, y](int n) {
                                return x < n && n < y;
                            }),
             vi.end());

    for (auto i: vi) {
        std::cout << i << " " ;    // 输出：5 28 590 245 24
    }

    std::cout << std::endl;

    return 0;
}
