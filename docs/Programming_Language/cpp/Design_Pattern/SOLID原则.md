# SOLID原则

## 单一职责原则 (SRP)

违反 SRP 原则：

```C++
#include <iostream>
#include <string>

// 定义 Customer 类
class Customer {
private:
    std::string name;

public:
    // 构造函数
    Customer(const std::string& customerName) : name(customerName) {}

    // 返回用户名
    const std::string& getName() const {
        return name;
    }

    void storeCustomer(const std::string& customerName) {
        std::cout << "Storing customer '" << customerName << "' into database..." << std::endl;
        // 存储客户到数据库...
    }

    void generateCustomerReport(const std::string& customerName) {
        std::cout << "Generating report for customer '" << customerName << "'..." << std::endl;
        // 生成客户报告...
    }
};
```
storeCustomer(const std::string& customerName) 职责是把顾客存入数据库。这个职责是持续的，应该把它放在顾客类的外面。
generateCustomerReport(const std::string& customerName) 职责是生成一个关于顾客的报告，所以它也应该放在顾客类的外面。

当一个类有多个职责，它就更加难以被理解，扩展和修改。

## 参考引用

