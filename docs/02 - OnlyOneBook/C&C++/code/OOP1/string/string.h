#ifndef __MYSTRING__
#define __MYSTRING__

class String {
    public:
        String(const char* cstr = nullptr);        // 默认构造函数，默认实参为nullptr
        String(const String &str);                 // 拷贝构造函数
        String &operator=(const String &str);      // 拷贝赋值函数
        ~String();                                 // 析构函数
        
        // 获取字符串，const 修饰成员函数，确保不会修改数据成员
        char* get_c_str() const {
            return m_data;
        }

    private:
        char* m_data;
};

#include <cstring>

inline String::String(const char* cstr) {
    if (cstr) {
        m_data = new char[strlen(cstr) + 1];
        strcpy(m_data, cstr);
    } else {
        // 分配一个字节的空间，用于存放空字符 '\0'
        m_data  = new char[1];
        *m_data = '\0';
    }
}

inline String::~String() {
    delete[] m_data;
}

inline String &String::operator=(const String &str) {
    if (this == &str)
        return *this;

    delete[] m_data;
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);
    return *this;
}

inline String::String(const String &str) {
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);
}

#include <iostream>
using namespace std;

ostream &operator<<(ostream &os, const String &str) {
    os << str.get_c_str();
    return os;
}

#endif
