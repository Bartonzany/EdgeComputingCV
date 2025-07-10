#include <iostream>
#include <vector>
#include <list>
#include <deque>
#include <iterator>

// 假设 SIZE 是一个常量，用于控制插入元素的数量
constexpr size_t SIZE = 10;    

// 泛型函数，用于测试容器的移动语义
template<typename Container>
void test_moveable(Container c) {
    // 使用迭代器类型萃取获取容器的值类型
    using ValueType = typename std::iterator_traits<typename Container::iterator>::value_type;

    // 向容器中插入 SIZE 个默认构造的元素
    for (size_t i = 0; i < SIZE; ++i) {
        c.insert(c.end(), ValueType());
    }

    // 输出容器的第一个元素（假设容器非空）
    if (!c.empty()) {
        std::cout << "First element: " << *(c.begin()) << std::endl;
    }

    // 测试容器的拷贝和移动语义
    Container c1(c);               // 拷贝构造
    Container c2(std::move(c));    // 移动构造
    c1.swap(c2);                   // 交换内容
}

int main() {
    test_moveable(std::list<int>{ 1, 2, 3 });
    test_moveable(std::vector<int>{ 4, 5, 6 });
    test_moveable(std::deque<int>{ 7, 8, 9 });

    return 0;
}