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

    b->show();     // 输出: Derived1 class show function
    d2->show();    // 输出: Derived2 class show function
    b2->show();    // 输出: Derived2 class show function

    delete b;
    delete d2;

    return 0;
}