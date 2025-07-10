#include <iostream>
#include <vector>
#include <map>

using namespace std;

// 函数声明
int  area(int w, int h);
void print();

// Rect 结构体定义
struct Rect {
        int width, height;
        int left, top;
        int (*getArea)(int, int);
        void (*printRect)();

        // 构造函数（默认）
        Rect(int w, int h, int l, int t):
            width(w), height(h), left(l), top(t), getArea(&area), printRect(&print) {}
};

// 成员函数实现
int area(int w, int h) {
    return w * h;
}

void print() {
    cout << "Printing rectangle..." << endl;
}

int main() {
    // Rect 初始化示例
    Rect r1 = { 3, 7, 20, 25 };    // 统一初始化
    Rect r2(3, 7, 20, 25);         // 使用构造函数

    // 数组初始化
    int ia[6] = { 27, 210, 12, 47, 109, 83 };

    // 使用 {} 统一初始化
    int                        x{ 10 };                            // 初始化 int
    double                     y{ 3.14 };                          // 初始化 double
    int                        arr[]{ 1, 2, 3 };                   // 初始化 array
    std::vector<int>           v{ 1, 2, 3 };                       // 初始化 vector
    std::map<int, std::string> m{ { 1, "one" }, { 2, "two" } };    // 初始化 map

    // 结构体初始化
    struct Point {
            int x, y;
    };
    Point p{ 1, 2 };    // 初始化结构体

    // 错误：窄化转换（取消注释会报错）
    // int x{3.14};  // 编译错误: narrowing conversion

    // 隐式转换允许
    int y_val(3.14);    // 允许：隐式转换，结果为 3

    // vector 初始化歧义说明
    std::vector<int> v1{ 1, 2, 3 };    // 使用 initializer_list，3 个元素：1, 2, 3
    std::vector<int> v2{ 10, 20 };     // 使用 initializer_list，2 个元素：10, 20
    std::vector<int> v3(10, 20);       // 使用构造函数，10 个元素，每个值为 20

    // 打印部分数据验证
    cout << "r1 area: " << r1.getArea(r1.width, r1.height) << endl;
    cout << "r2 area: " << r2.getArea(r2.width, r2.height) << endl;
    cout << "ia size: " <<  sizeof(ia) / sizeof(ia[0]) << endl;
    cout << "x = " << x << ", y = " << y << endl;
    cout << "arr[0] = " << arr[0] << ", arr[1] = " << arr[1] << endl;
    cout << "Point p: (" << p.x << ", " << p.y << ")" << endl;
    cout << "y_val = " << y_val << " (" << typeid(y_val).name() << ")" << endl;
    cout << "v1 size: " << v1.size() << ", v2 size: " << v2.size() << ", v3 size: " << v3.size() << endl;

    return 0;
}