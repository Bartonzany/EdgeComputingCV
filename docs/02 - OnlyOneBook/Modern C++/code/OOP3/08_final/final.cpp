#include <iostream>

// 1. 类的 final：禁止继承
struct Base1 final {
        void func() {
            std::cout << "Base1::func()" << std::endl;
        }
};

// struct Derived1 : Base1 {}; // 编译错误：Base1 是 final，不能被继承

// 2. 虚函数的 final：禁止重写
struct Base2 {
        virtual void f() final {
            std::cout << "Base2::f()" << std::endl;
        }
};

struct Derived2: Base2 {
        // void f() override {} // 编译错误：Base2::f 是 final，不能被重写
};

struct Base3 {
        virtual void g() {}
};

struct Derived3: Base3 {
        void g() override final {    // 禁止进一步重写
            std::cout << "Derived3::g()" << std::endl;
        }
};

int main() {
    Base1 b1;
    b1.func();    // 输出: Base1::func()

    Base2 b2;
    b2.f();    // 输出: Base2::f()

    Derived2 d2;
    d2.f();    // 输出: Base2::f()

    Base3* ptr;

    Derived3 d3;
    ptr = &d3;
    ptr->g();    // 输出: Derived3::g()

    return 0;
}