#include <iostream>
#include <map>
#include <string>
#include <set>

// 示例 1：使用 decltype 声明变量类型
std::map<std::string, float> coll;
decltype(coll)::value_type   elem;    // 推导为 std::pair<const std::string, float>

// 示例 2：模板函数中使用 decltype 推导返回值类型
template<typename T1, typename T2>
auto add(T1 x, T2 y) -> decltype(x + y) { // C++14 可以直接写为 auto add(T1 x, T2 y)
    return x + y;
}

// 示例 3：模板元编程中使用 decltype 获取类型信息
template<typename T>
void test_decltype(const T &obj) {
    decltype(obj) anotherObj = obj;    // 复制对象
    std::cout << "Container has size: " << anotherObj.size() << std::endl;

    for (const auto &pair: anotherObj) {
        std::cout << pair.first << " -> " << pair.second << std::endl;
    }
}

// 示例 4：使用 decltype 推导 lambda 表达式的类型
struct Person {
        std::string firstname;
        std::string lastname;
};

int main() {
    // 示例 1 使用
    auto elem = std::make_pair("pi", 3.14f);    // 初始化 pair
    std::cout << "Example 1 - Pair value: " << elem.first << " => " << elem.second << "\n";

    // 示例 2 使用
    auto result1 = add(3, 4);                              // int
    auto result2 = add(3.5, 4.2);                          // double
    auto result3 = add(std::string("Hello "), "World");    // string + const char*
    std::cout << "Example 2 - Results: " << result1 << ", " << result2 << ", " << result3 << "\n";

    // 示例 3 使用
    std::cout << "Example 3 - Map print"
              << "\n";
    std::map<int, std::string> myMap{ { 1, "one" }, { 2, "two" } };
    test_decltype(myMap);

    // 示例 4 使用
    auto cmp = [](const Person &p1, const Person &p2) {
        return p1.lastname < p2.lastname || (p1.lastname == p2.lastname && p1.firstname < p2.firstname);
    };

    std::set<Person, decltype(cmp)> coll(cmp);
    coll.insert(Person{ "John", "Doe" });
    coll.insert(Person{ "Jane", "Smith" });
    coll.insert(Person{ "Alice", "Doe" });

    std::cout << "Example 4 - Sorted Persons:\n";
    for (const auto &person: coll) {
        std::cout << person.firstname << " " << person.lastname << std::endl;
    }

    return 0;
}