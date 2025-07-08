#include "complex.h"
#include <iostream>

using namespace std;

// 重载 << 运算符以输出复数
ostream &operator<<(ostream &os, const complex &x) {
    return os << '(' << real(x) << ',' << imag(x) << ')';
}

int main() {
    // 定义两个复数
    complex c1(2, 1);
    complex c2(4, 0);

    // 输出复数 c1 和 c2
    cout << "c1: " << c1 << endl;
    cout << "c2: " << c2 << endl;

    // 复数运算
    cout << "c1 + c2: " << c1 + c2 << endl;
    cout << "c1 - c2: " << c1 - c2 << endl;
    cout << "c1 * c2: " << c1 * c2 << endl;
    cout << "c1 / 2: " << c1 / 2 << endl;

    // 复数的共轭、模和极坐标表示
    cout << "conj(c1): " << conj(c1) << endl;
    cout << "norm(c1): " << norm(c1) << endl;
    cout << "polar(10, 4): " << polar(10, 4) << endl;

    // 复合赋值运算
    cout << "c1 += c2: " << (c1 += c2) << endl;

    // 比较运算
    cout << "c1 == c2: " << (c1 == c2) << endl;
    cout << "c1 != c2: " << (c1 != c2) << endl;

    // 一元运算
    cout << "+c2: " << +c2 << endl;
    cout << "-c2: " << -c2 << endl;

    // 复数与标量的运算
    cout << "c2 - 2: " << (c2 - 2) << endl;
    cout << "5 + c2: " << (5 + c2) << endl;

    return 0;
}
