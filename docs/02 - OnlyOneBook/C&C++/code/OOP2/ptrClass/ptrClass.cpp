#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Student {
public:
    Student(const string& name, int age) : name(name), age(age) {
        cout << "Student " << name << " created." << endl;
    }

    ~Student() {
        cout << "Student " << name << " destroyed." << endl;
    }

    void display() const {
        cout << "Name: " << name << ", Age: " << age << endl;
    }

private:
    string name;
    int age;
};

int main() {
    shared_ptr<Student> studentPtr(new Student("Alice", 20));
    
    studentPtr->display();

    // 共享指针
    shared_ptr<Student> studentPtr2 = studentPtr;

    // 输出学生信息
    studentPtr2->display();

    cout << "use count: " << studentPtr.use_count() << endl;

    // 在作用域结束时，共享指针会自动释放资源
    return 0;
}
