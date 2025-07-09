#include <iostream>
#include <vector>
#include <list>
#include <deque>

// =============================
// 1. 普通模板参数版本
// =============================
template<typename T, typename Container = std::vector<T>>
class MyStack {
    public:
        void push(const T &value) {
            data.push_back(value);
        }
        void pop() {
            data.pop_back();
        }
        const T &top() const {
            return data.back();
        }
        bool empty() const {
            return data.empty();
        }

    private:
        Container data;
};

// =============================
// 2. 模板模板参数版本
// =============================
template<
    typename T,
    template<typename, typename> class Container = std::vector>
class MyTTStack {
    public:
        void push(const T &value) {
            data.push_back(value);
        }
        void pop() {
            data.pop_back();
        }
        const T &top() const {
            return data.back();
        }
        bool empty() const {
            return data.empty();
        }

    private:
        Container<T, std::allocator<T>> data;
};

int main() {
    // ===== 使用普通模板参数 =====
    MyStack<int> stack1;    // 默认使用 vector<int>
    stack1.push(1);
    stack1.push(2);
    std::cout << "stack1.top() = " << stack1.top() << "\n";    // 输出 2

    MyStack<std::string, std::list<std::string>> stack2;
    stack2.push("Hello");
    stack2.push("World");
    std::cout << "stack2.top() = " << stack2.top() << "\n";    // 输出 World

    // ===== 使用模板模板参数 =====
    MyTTStack<double> stack3;    // 使用 vector<double>
    stack3.push(3.14);
    stack3.push(2.71);
    std::cout << "stack3.top() = " << stack3.top() << "\n";    // 输出 2.71

    MyTTStack<char, std::list> stack4;    // 使用 list<char>
    stack4.push('A');
    stack4.push('B');
    std::cout << "stack4.top() = " << stack4.top() << "\n";    // 输出 B

    return 0;
}