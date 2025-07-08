#include "string.h"
#include <iostream>

using namespace std;

int main() {
    String s1("hello");
    String s2("world");

    String s3(s2);        // 调用拷贝构造函数
    cout << s3 << endl;

    s3 = s1;              // 调用拷贝赋值函数
    cout << s3 << endl;
    cout << s2 << endl;
    cout << s1 << endl;
}
