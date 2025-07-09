#include <iostream>

// 类模板：复数类
template<typename T>
class complex {
    public:
        complex(T r = 0, T i = 0):
            re(r), im(i) {}

        complex &operator+=(const complex &rhs) {
            re += rhs.re;
            im += rhs.im;
            return *this;
        }

        T real() const {
            return re;
        }
        T imag() const {
            return im;
        }

    private:
        T re, im;
};

// 函数模板：求最小值
template<class T>
inline const T &min(const T &a, const T &b) {
    return b < a ? b : a;
}

// 类模板：pair
template<class T1, class T2>
struct pair {
        typedef T1 first_type;
        typedef T2 second_type;

        T1 first;
        T2 second;

        // 默认构造函数
        pair():
            first(T1()), second(T2()) {}

        // 带参数构造函数
        pair(const T1 &a, const T2 &b):
            first(a), second(b) {}

        // 成员模板构造函数：允许不同类型之间的转换
        template<class U1, class U2>
        pair(const pair<U1, U2> &p):
            first(p.first), second(p.second) {}
};

int main() {
    // 使用 complex 类模板
    complex<double> c1(3.0, 4.0);
    complex<double> c2(1.0, 2.0);
    c1 += c2;

    std::cout << "c1 = (" << c1.real() << ", " << c1.imag() << "i)" << std::endl;

    // 使用函数模板 min
    int a = 5, b = 9;
    std::cout << "min(a, b) = " << min(a, b) << std::endl;

    // 使用 pair 类模板和成员模板构造函数
    pair<int, double> p1(42, 3.14);
    pair<long, float> p2 = p1;    // 成员模板允许不同类型的转换
    std::cout << "p2 = (" << p2.first << ", " << p2.second << ")" << std::endl;

    return 0;
}
