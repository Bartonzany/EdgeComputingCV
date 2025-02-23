// 防卫式声明：通过 #ifndef、#define、#endif防止头文件被重复包含
// 现代C++可使用 #pragma once替代（编译器扩展，非标准但广泛支持）
// 编译器在识别到 #pragma once后直接跳过后续内容，比传统宏防卫更快
#pragma once
#ifndef __MYCOMPLEX__
#define __MYCOMPLEX__

class complex;
complex &__doapl(complex* ths, const complex &r);
complex &__doami(complex* ths, const complex &r);
complex &__doaml(complex* ths, const complex &r);

class complex {
    public:
        // 构造函数可以使用默认实参和成员初始化列表
        // 在初始值列表中，才是初始化，在构造函数体内的，叫做赋值
        // 传引用的写法也可以：complex(double &r = 0, double &i = 0) : re(r), im(i) {}
        complex(double r = 0, double i = 0):
            re(r), im(i) {}

        // 函数重载，设计为成员函数，在类声明内定义的函数会自动成为inline函数
        complex &operator+=(const complex &);
        complex &operator-=(const complex &);
        complex &operator*=(const complex &);
        complex &operator/=(const complex &);

        // 对于不改变类属性（数据成员）的成员函数，务必加上 const 声明
        double real() const {
            return re;
        }
        double imag() const {
            return im;
        }

    private:
        // 将数据成员放在private声明下，提供接口函数访问数据，从而保证数据的安全性
        double re, im;

        // 友元函数不受访问控制的限制，可以自由访问类中所有成员，包括私有成员和保护成员
        friend complex &__doapl(complex*, const complex &);
        friend complex &__doami(complex*, const complex &);
        friend complex &__doaml(complex*, const complex &);
};

// 在类声明外定义的函数需要显式加上inline关键字才能成为inline函数
// this指针所指向的内容可能会在函数内部被修改，所以不加const
inline complex &__doapl(complex* ths, const complex &r) {
    ths->re += r.re;
    ths->im += r.im;
    return *ths;
}

// 参数使用const关键字进行修饰，以确保在函数内部该参数不会被修改
// 函数的返回值是一个已经存在的对象，所以传引用
inline complex &complex::operator+=(const complex &r) {
    return __doapl(this, r);
}

inline complex &__doami(complex* ths, const complex &r) {
    ths->re -= r.re;
    ths->im -= r.im;
    return *ths;
}

inline complex &complex::operator-=(const complex &r) {
    return __doami(this, r);
}

inline complex &__doaml(complex* ths, const complex &r) {
    double f = ths->re * r.re - ths->im * r.im;
    ths->im  = ths->re * r.im + ths->im * r.re;
    ths->re  = f;
    return *ths;
}

inline complex &complex::operator*=(const complex &r) {
    return __doaml(this, r);
}

inline double imag(const complex &x) {
    return x.imag();
}

inline double real(const complex &x) {
    return x.real();
}

// 参数传引用，函数返回的是局部对象（local object）所以传值
inline complex operator+(const complex &x, const complex &y) {
    return complex(real(x) + real(y), imag(x) + imag(y));
}

// 如果 y 是一个常量或临时值，传引用可能会导致编译错误或未定义行为，所以传值
inline complex operator+(const complex &x, double y) {
    return complex(real(x) + y, imag(x));
}

inline complex operator+(double x, const complex &y) {
    return complex(x + real(y), imag(y));
}

inline complex operator-(const complex &x, const complex &y) {
    return complex(real(x) - real(y), imag(x) - imag(y));
}

inline complex operator-(const complex &x, double y) {
    return complex(real(x) - y, imag(x));
}

inline complex operator-(double x, const complex &y) {
    return complex(x - real(y), -imag(y));
}

inline complex operator*(const complex &x, const complex &y) {
    return complex(real(x) * real(y) - imag(x) * imag(y),
                   real(x) * imag(y) + imag(x) * real(y));
}

inline complex operator*(const complex &x, double y) {
    return complex(real(x) * y, imag(x) * y);
}

inline complex operator*(double x, const complex &y) {
    return complex(x * real(y), x * imag(y));
}

complex operator/(const complex &x, double y) {
    return complex(real(x) / y, imag(x) / y);
}

inline complex operator+(const complex &x) {
    return x;
}

inline complex operator-(const complex &x) {
    return complex(-real(x), -imag(x));
}

inline bool operator==(const complex &x, const complex &y) {
    return real(x) == real(y) && imag(x) == imag(y);
}

inline bool operator==(const complex &x, double y) {
    return real(x) == y && imag(x) == 0;
}

inline bool operator==(double x, const complex &y) {
    return x == real(y) && imag(y) == 0;
}

inline bool operator!=(const complex &x, const complex &y) {
    return real(x) != real(y) || imag(x) != imag(y);
}

inline bool operator!=(const complex &x, double y) {
    return real(x) != y || imag(x) != 0;
}

inline bool operator!=(double x, const complex &y) {
    return x != real(y) || imag(y) != 0;
}

#include <cmath>

inline complex polar(double r, double t) {
    return complex(r * cos(t), r * sin(t));
}

inline complex conj(const complex &x) {
    return complex(real(x), -imag(x));
}

inline double norm(const complex &x) {
    return real(x) * real(x) + imag(x) * imag(x);
}

#endif    //__MYCOMPLEX__
