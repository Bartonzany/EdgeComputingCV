#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 仿函数：用于统计操作次数
class Counter {
public:
    Counter() : count(0) {}

    void operator()(int value) {
        cout << "Processing value: " << value << endl;
        count++;
    }

    int getCount() const {
        return count;
    }

private:
    int count;
};

// 仿函数：自定义排序规则
class GreaterThan {
public:
    GreaterThan(int threshold) : threshold(threshold) {}

    bool operator()(int value) const {
        return value > threshold;
    }

private:
    int threshold;
};

int main() {
    vector<int> numbers = {10, 5, 20, 15, 30};

    // 应用1：统计操作次数
    Counter counter;
    counter = for_each(numbers.begin(), numbers.end(), counter);
    cout << "Total processed values: " << counter.getCount() << endl;

    // 应用2：自定义排序规则
    GreaterThan isGreaterThan(15);
    auto it = find_if(numbers.begin(), numbers.end(), isGreaterThan);
    if (it != numbers.end()) {
        cout << "First value greater than 15: " << *it << endl;
    }

    // 应用3：作为回调函数
    auto callback = [](int value) {
        cout << "Callback executed with value: " << value << endl;
    };
    for_each(numbers.begin(), numbers.end(), callback);

    return 0;
}
