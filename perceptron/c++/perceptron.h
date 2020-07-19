#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include<iostream>
#include<vector>
#include<string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

class perceptron
{
    // private 说明符之后的成员可以被类的成员函数访问，但是不能被使用该类的代码访问
    private:
    vector<double> w;
    double b;
    vector<vector<double>> X_train;
    vector<vector<double>> X_test;
    vector<int> y_train;
    vector<int> y_test;
    // 定义在public说明符之后的在整个程序内可被访问
    public:
    perceptron(); // 构造函数
    // virtual 说明符指定非静态成员函数为虚函数并支持动态调用派发
    // https://zh.cppreference.com/w/cpp/language/virtual
    vector<vector<double>> load_mnist(const string& file_name);
    void splitData(vector<vector<double>>& dataArr, const double& train_rate);
    void train(const int& step,const float& lr);
    double test();
};
#endif

