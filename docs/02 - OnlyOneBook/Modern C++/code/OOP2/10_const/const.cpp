#include <iostream>

using namespace std;

class MyClass {
    public:
        MyClass():
            value(0) {}

        // 非常量成员函数
        void setValue(int newValue) {
            value = newValue;
        }

        // 常量成员函数
        int getValue() const {
            return value;
        }

        // 常量成员函数和非常量成员函数重载
        void print() const {
            cout << "Const function: " << value << endl;
        }

        void print() {
            cout << "Non-const function: " << value << endl;
        }

    private:
        int value;
};

int main() {
    MyClass       obj;
    const MyClass constObj;

    // 非常量对象可以调用非常量成员函数和常量成员函数
    obj.setValue(10);
    cout << "Value from non-const object: " << obj.getValue() << endl;

    // 常量对象只能调用常量成员函数
    // constObj.setValue(20); // 错误：常量对象不能调用非常量成员函数
    cout << "Value from const object: " << constObj.getValue() << endl;

    // 常量对象调用常量成员函数
    constObj.print();    // 输出: Const function: 0

    // 非常量对象调用非常量成员函数
    obj.print();    // 输出: Non-const function: 10

    return 0;
}
