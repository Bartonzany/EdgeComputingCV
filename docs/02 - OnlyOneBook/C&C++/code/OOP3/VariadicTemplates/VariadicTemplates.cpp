#include <iostream>
#include <bitset>

using namespace std;

void print() {
    cout << "End of recursion" << std::endl;
}

template<typename T, typename... Types>
void print(const T &firstArg, const Types &... args) {
    cout << "Number of types: " << sizeof...(args) << endl;    // 打印参数包的参数个数
    cout << firstArg << endl;
    print(args...);    // 递归调用
}

int main() {
    print(7.5, 42, 2.5, "Hello", 'A', bitset<16>(377));
    return 0;
}