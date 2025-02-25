#include <iostream>
using namespace std;

class Base {
    public:
        virtual void vfunc1() {
            cout << "Base::vfunc1" << endl;
        }
        virtual ~Base() {
            cout << "Base destructor" << endl;
        }
};

class Derived1: public Base {
    public:
        virtual void vfunc1() {    // 使用 virtual 重写
            cout << "Derived1::vfunc1" << endl;
        }
        ~Derived1() {
            cout << "Derived1 destructor" << endl;
        }
};

class Derived2: public Base {
    public:
        void vfunc1() override {    // 使用 override 重写
            cout << "Derived2::vfunc1" << endl;
        }
        ~Derived2() {
            cout << "Derived2 destructor" << endl;
        }
};

int main() {
    Base* obj1 = new Derived1();
    Base* obj2 = new Derived2();
    obj1->vfunc1();    // 输出: Derived1::vfunc1
    obj2->vfunc1();    // 输出: Derived2::vfunc1
    delete obj1;       // 输出: Derived1 destructor Base destructor
    delete obj2;       // 输出：Derived2 destructor Base destructor
    return 0;
}
