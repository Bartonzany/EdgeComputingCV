#include <iostream>

using namespace std;

class Base {
    public:
        virtual void show() {
            cout << "Base class show function" << endl;
        }
        virtual ~Base() {
            cout << "Base class destructor" << endl;
        }
};

class Derived1: public Base {
    public:
        virtual void show() {
            cout << "Derived1 class show function" << endl;
        }
};

class Derived2: public Base {
    public:
        virtual void show() {
            cout << "Derived2 class show function" << endl;
        }
};

int main() {
    Base*     b  = new Derived1();
    Derived2* d2 = new Derived2();
    Base*     b2 = d2;
    Base*     b3 = new Derived1();

    b->show();           // 动态绑定，输出 Derived1 class show function
    d2->show();          // 动态绑定，输出 Derived2 class show function
    b2->show();          // 动态绑定，输出 Derived2 class show function
    b3->Base::show();    // 静态绑定，输出 Base class show function

    delete b;
    delete d2;
    delete b3;

    return 0;
}