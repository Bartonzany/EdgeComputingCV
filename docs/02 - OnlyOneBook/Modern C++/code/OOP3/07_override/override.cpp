#include <iostream>

struct Base {
        virtual void vfunc(float) {
            std::cout << "Base::vfunc(float)" << std::endl;
        }
};

// 错误示例：未使用 override，导致签名错误（不会覆盖父类函数）
struct Derived1: Base {
        virtual void vfunc(int) {    // 参数类型不同，无法覆盖 Base::vfunc(float)
            std::cout << "Derived1::vfunc(int)" << std::endl;
        }
};

// 正确示例：使用 override 捕获签名错误
struct Derived2: Base {
        // virtual void vfunc(int) override {    // 编译器会报错：没有匹配的虚函数
        //     std::cout << "Derived2::vfunc(int)" << std::endl;
        // }
};

// 正确示例：正确重写父类虚函数
struct Derived3: Base {
        virtual void vfunc(float) override {    // 正确重写，编译通过
            std::cout << "Derived3::vfunc(float)" << std::endl;
        }
};

int main() {
    Base     b;
    Derived1 d1;
    Derived3 d3;

    // Base 调用
    b.vfunc(3.14f);    // 输出: Base::vfunc(float)

    // Derived1 调用
    d1.vfunc(42);    // 输出: Derived1::vfunc(int)

    // Derived3 调用
    d3.vfunc(3.14f);    // 输出: Derived3::vfunc(float)

    return 0;
}