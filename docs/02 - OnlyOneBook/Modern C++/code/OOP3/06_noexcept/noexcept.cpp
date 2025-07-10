#include <iostream>
#include <cstring>
#include <utility>

// 声明一个不会抛出异常的函数
void foo() noexcept {
    std::cout << "foo() called\n";
}

// 条件性 noexcept：只有当 x.swap(y) 不抛出异常时，swap 函数也不会抛出异常
template<typename T>
void swap(T &x, T &y) noexcept(noexcept(x.swap(y))) {
    x.swap(y);
}

// 自定义类 MyString，支持移动语义
class MyString {
    private:
        char*  data;
        size_t len;

    public:
        // 默认构造函数
        MyString(const char* str = ""):
            len(std::strlen(str)) {
            data = new char[len + 1];
            std::strcpy(data, str);
        }

        // 析构函数
        ~MyString() {
            delete[] data;
        }

        // 拷贝构造函数
        MyString(const MyString &str):
            len(str.len) {
            data = new char[len + 1];
            std::strcpy(data, str.data);
        }

        // 拷贝赋值运算符
        MyString &operator=(const MyString &str) {
            if (this != &str) {
                delete[] data;
                len  = str.len;
                data = new char[len + 1];
                std::strcpy(data, str.data);
            }
            return *this;
        }

        // 移动构造函数
        MyString(MyString &&str) noexcept:
            data(str.data), len(str.len) {
            str.data = nullptr;
            str.len  = 0;
        }

        // 移动赋值运算符
        MyString &operator=(MyString &&str) noexcept {
            if (this != &str) {
                delete[] data;
                data     = str.data;
                len      = str.len;
                str.data = nullptr;
                str.len  = 0;
            }
            return *this;
        }

        // swap 成员函数
        void swap(MyString &other) noexcept {
            std::swap(data, other.data);
            std::swap(len, other.len);
        }

        // 打印字符串内容
        void print() const {
            if (data)
                std::cout << data;
            else
                std::cout << "(null)";
        }
};

int main() {
    // 测试 foo()
    foo();

    // 测试 MyString 的移动构造和移动赋值
    MyString s1("Hello");
    MyString s2 = std::move(s1);    // 调用移动构造函数
    MyString s3("World");
    s3 = std::move(s2);    // 调用移动赋值运算符

    std::cout << "s3: ";
    s3.print();
    std::cout << std::endl;

    // 测试自定义 swap
    MyString a("ABC");
    MyString b("XYZ");
    ::swap(a, b);    // 使用我们定义的 swap 版本

    std::cout << "After swap:\n";
    std::cout << "a: ";
    a.print();
    std::cout << "\nb: ";
    b.print();
    std::cout << std::endl;

    return 0;
}