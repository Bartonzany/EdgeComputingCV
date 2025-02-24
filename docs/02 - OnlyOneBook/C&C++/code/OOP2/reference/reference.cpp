#include <iostream>
using namespace std;

typedef struct Stag {
        int a, b, c, d;
} S;

int main() {
    double  x = 0;
    double* p = &x;    // p 指向 x，p 的值是 x 的地址
    double &r = x;     // r 代表 x，现在 r 和 x 都是 0

    cout << "Size of x (double): " << sizeof(x) << endl;                 // 8 (double 的大小)
    cout << "Size of p (pointer to double): " << sizeof(p) << endl;      // 4 (指针的大小，通常是 4 字节)
    cout << "Size of r (reference to double): " << sizeof(r) << endl;    // 8 (引用 r 的大小与 x 相同)

    cout << "Address of x (p): " << p << endl;          // 0065FDFC (x 的地址)
    cout << "Value of x through p: " << *p << endl;     // 0 (通过指针访问 x 的值)
    cout << "Value of x directly: " << x << endl;       // 0 (直接访问 x 的值)
    cout << "Value of x through r: " << r << endl;      // 0 (通过引用访问 x 的值)
    cout << "Address of x directly: " << &x << endl;    // 0065FDFC (x 的地址)
    cout << "Address of r: " << &r << endl;             // 0065FDFC (引用 r 的地址与 x 相同)

    S  s;
    S &rs = s;
    cout << "Size of s (struct S): " << sizeof(s) << endl;                   // 16 (结构体 S 的大小)
    cout << "Size of rs (reference to struct S): " << sizeof(rs) << endl;    // 16 (引用 rs 的大小与 s 相同)
    cout << "Address of s: " << &s << endl;                                  // 0065FDE8 (s 的地址)
    cout << "Address of rs: " << &rs << endl;                                // 0065FDE8 (引用 rs 的地址与 s 相同)

    return 0;
}
