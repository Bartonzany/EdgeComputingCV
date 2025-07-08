#include <iostream>

class Fraction {
    public:
        // 使用explicit声明构造函数，防止隐式类型转换
        // explicit Fraction(int num, int den = 1)		
        //         : m_numerator(num), m_denominator(den) {}
        Fraction(int num, int den = 1)		
                : m_numerator(num), m_denominator(den) {}
        
        // 使用explicit声明类型转换运算符，防止隐式类型转换
        explicit operator double() const {    		
            return (double) (m_numerator * 1.0 / m_denominator);
        }

        double operator+(double d) const {
            return (double) (m_numerator * 1.0 / m_denominator) + d;
        }

        Fraction operator+(const Fraction &f) const {
            return Fraction(m_numerator + f.m_numerator, m_denominator + f.m_denominator);
        }
    
    private:
        int m_numerator;        
        int m_denominator;      
    };

int main() {
    Fraction f1(3, 5);
    double d = f1 + 4; 	// 正确，显式类型转换
    Fraction f2 = f1 + 4;
    printf("f2 = %f\n", (double)f2);
    printf("%f\n", d);
    return 0;
}