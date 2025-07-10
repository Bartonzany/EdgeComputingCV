#include <iostream>
#include <vector>
#include <list>
#include <deque>

constexpr size_t SIZE = 10;    // 假设 SIZE 是一个常量，用于控制插入元素的数量

// 定义一个模板别名，简化容器的声明
template<typename T>
using MyContainer = std::vector<T>;

// 模板模板参数示例
template<typename T, template<typename> class Container>
class XCls {
    private:
        Container<T> c;    // 使用模板模板参数定义的容器

    public:
        // 构造函数：向容器中插入 SIZE 个默认构造的元素
        XCls() {
            for (size_t i = 0; i < SIZE; ++i) {
                c.insert(c.end(), T());
            }
        }

        XCls(const std::initializer_list<T> &initList) {
            for (const auto &val: initList) {
                c.insert(c.end(), val);
            }
        }

        // 输出容器的第一个元素（假设容器非空）
        void printFirstElement() const {
            if (!c.empty()) {
                std::cout << "First element: " << *(c.begin()) << std::endl;
            } else {
                std::cout << "Container is empty!" << std::endl;
            }
        }

        // 测试容器的拷贝和移动语义
        void testMoveSemantics() {
            Container<T> c1(c);               // 拷贝构造
            Container<T> c2(std::move(c));    // 移动构造
            c1.swap(c2);                      // 交换内容
        }
};

int main() {
    // 使用初始化列表构造对象
    XCls<int, MyContainer> obj1({ 1, 2, 3, 4, 5 });
    obj1.printFirstElement();
    obj1.testMoveSemantics();

    // 使用默认构造函数
    XCls<int, MyContainer> obj2;
    obj2.printFirstElement();
    obj2.testMoveSemantics();

    return 0;
}