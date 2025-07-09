#include <iostream>
#include <typeinfo>

// 主模板：泛型版本
template<typename T>
struct TypePrinter {
        static void print() {
            std::cout << "Generic type: " << typeid(T).name() << std::endl;
        }
};

// 模板全特化：int 类型
template<>
struct TypePrinter<int> {
        static void print() {
            std::cout << "Type is int" << std::endl;
        }
};

// 模板偏特化：指针类型
template<typename T>
struct TypePrinter<T*> {
        static void print() {
            std::cout << "Pointer to type: " << typeid(T).name() << std::endl;
        }
};

// 偏特化：数组类型
template<typename T, size_t N>
struct TypePrinter<T[N]> {
        static void print() {
            std::cout << "Array of " << N << " elements of type: " << typeid(T).name() << std::endl;
        }
};

int main() {
    TypePrinter<double>::print();      // 泛型
    TypePrinter<int>::print();         // 全特化
    TypePrinter<char*>::print();       // 偏特化：指针
    TypePrinter<float[5]>::print();    // 偏特化：数组

    return 0;
}
