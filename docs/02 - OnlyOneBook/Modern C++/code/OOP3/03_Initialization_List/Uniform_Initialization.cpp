#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <initializer_list>

using namespace std;

// 示例类 P：演示 initializer_list 构造函数与普通构造函数的区别
class P {
    public:
        // 普通构造函数
        P(int a, int b) {
            cout << "P(int, int), a = " << a << ", b = " << b << endl;
        }

        // 使用 initializer_list 的构造函数（支持任意数量的 int 参数）
        P(initializer_list<int> initlist) {
            cout << "P(std::initializer_list<int>), values = ";
            for (auto i: initlist) {
                cout << i << " ";
            }
            cout << endl;
        }
};

int main() {
    cout << "=== 测试类 P 的构造函数调用 ===" << endl;
    P p(77, 5);          // 调用普通构造函数
    P q{ 77, 5 };        // 调用 initializer_list 构造函数
    P r{ 77, 5, 42 };    // 多个参数也调用 initializer_list 构造函数
    P s = { 77, 5 };     // 等价于 q，也调用 initializer_list 构造函数
    (void)s;

    cout << "\n=== 测试 vector 初始化和插入操作 ===" << endl;
    // 使用 initializer_list 初始化 vector
    vector<int> v1{ 2, 5, 7, 13, 69, 83, 50 };
    vector<int> v2({ 2, 5, 7, 13, 69, 83, 50 });

    vector<int> v3;
    v3 = { 2, 5, 7, 13, 69, 83, 50 };    // 赋值时使用 initializer_list

    // 插入新的 initializer_list 到 vector 中
    v3.insert(v3.begin() + 2, { 0, 1, 2, 3, 4 });

    cout << "v3 内容: ";
    for (auto i: v3) {
        cout << i << " ";
    }
    cout << endl;

    cout << "\n=== 测试算法库中 initializer_list 的使用 ===" << endl;
    // 使用 initializer_list 作为参数传给 max/min
    cout << "max string: "
         << max({ string("Ace"), string("stacy"), string("sabrina"), string("Barkley") }) << endl;

    cout << "min string: "
         << min({ string("Ace"), string("stacy"), string("sabrina"), string("Barkley") }) << endl;

    cout << "max number: " << max({ 54, 16, 48, 5 }) << endl;
    cout << "min number: " << min({ 54, 16, 48, 5 }) << endl;

    return 0;
}