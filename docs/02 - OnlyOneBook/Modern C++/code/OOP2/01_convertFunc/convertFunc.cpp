#include <iostream>

class Fraction {
    public:
        // 函数1：使用explicit声明构造函数，防止隐式类型转换
        // explicit Fraction(int num, int den = 1)
        //         : m_numerator(num), m_denominator(den) {}
        Fraction(int num, int den = 1):
            m_numerator(num), m_denominator(den) {}

        // 函数2：重载运算符 +，用于 Fraction 对象之间的加法
        Fraction operator+(const Fraction &f) const {
            return Fraction(m_numerator * f.m_denominator + f.m_numerator * m_denominator,
                            m_denominator * f.m_denominator);
        }

        // 函数3：使用隐式类型转换，会与重载运算符 + 冲突
        // operator double() const {
        //     return (double)(m_numerator * 1.0 / m_denominator);
        // }

        // 函数4：使用explicit声明类型转换运算符，防止隐式类型转换
        explicit operator double() const {
            return (double)(m_numerator * 1.0 / m_denominator);
        }

    private:
        int m_numerator;
        int m_denominator;
};

int main() {
    Fraction f1(3, 5);
    Fraction f2 = f1 + 4;    // 调用函数2，即重载的运算符 +，将 Fraction 对象与整数相加
    // double d = f1 + 4;         // 调用函数3，即隐式类型转换，将 Fraction 对象转换为 double，编译歧义

    printf("f2 = %f\n", (double)f2);
    // printf("d = %f\n", d);
    return 0;
}