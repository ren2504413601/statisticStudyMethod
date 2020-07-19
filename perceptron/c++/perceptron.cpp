// #include<iostream>
// #include<vector>
// #include<string>
#include<algorithm>
#include<sstream>
#include<fstream> 
#include"perceptron.h"

// using std::cout;
// using std::endl;
// using std::vector;

perceptron::perceptron()
{} 

vector<vector<double>> perceptron::load_mnist(const string& file_name) // const引用（常量引用）
{
    std::cout<< "start loading!"<<endl;
    vector<vector<double>> dataArr;
    vector<double> tmpData;
    string line;
    double onepoint;
    char ch;
    // ifstream 定义于头文件 <fstream>
    // https://zh.cppreference.com/w/cpp/io/basic_ifstream
    std::ifstream infile(file_name);
    if (!infile) 
    {
        cout << "Open file error!" << endl;
        exit(0);
    }
    while (infile) // 检查是否到达了文件末尾
    {
        tmpData.clear();
        std::getline(infile,line);
        if(line.empty()) continue;
        // stringstream 定义于头文件 <sstream>
        std::stringstream stringin(line);
        // 释出浮点值，潜在地跳过前导空格。存储值到给定的引用 onepoint
        // https://zh.cppreference.com/w/cpp/io/basic_istream/operator_gtgt
        while(stringin){
            stringin >> onepoint;
            tmpData.push_back(onepoint);
            stringin >> ch;
        }
        // atof 转译 str 所指向的字节字符串中的浮点值
        dataArr.push_back(tmpData);
    }
    return dataArr;
}
void perceptron::splitData(vector<vector<double>>& dataArr, const double& train_rate)
{
    // random_shuffle 定义于头文件 <algorithm>
    std::random_shuffle(dataArr.begin(), dataArr.end());
    unsigned long size = dataArr.size();
    unsigned long trainSize = size * train_rate;
    std::cout<<"totoal data size ="<<size<<", training data size ="<< trainSize<<endl;
    for (int i = 0; i < trainSize; ++i)
    {
        vector<double> tmpData(dataArr[i]);
        if (tmpData.back() >= 5) y_train.push_back(1);
        else y_train.push_back(-1);
        tmpData.pop_back();
        X_train.push_back(tmpData);
    }
    for (int j = trainSize; j < size; ++j)
    {
        vector<double> tmpData(dataArr[j]);
        if (tmpData.back() >= 5) y_test.push_back(1);
        else y_test.push_back(-1);
        tmpData.pop_back();
        X_test.push_back(tmpData);
    }
}

void perceptron::train(const int& step,const float& lr)
{
    int m = X_train.size(), n = X_train[0].size();
    w = vector<double>(n, 0);
    b = 0;
    std::cout<<"Training start!"<<endl;
    for (int s = 0; s < step; ++s)
    {
        for (int i = 0; i < m; ++i)
        {
            vector<double> xi = X_train[i];
            double yi = y_train[i];
            double dotVal = 0.0;
            for (int j = 0; j < n; ++j)
            {
                dotVal += (w[j]*xi[j]);
            }
            if ((yi*(dotVal+b)) <= 0)
            {
                b += lr*yi;
                for (int jj = 0; jj < n; ++jj) w[jj] += (lr*yi*xi[jj]);
            }
        }
        std::cout<<"round: "<< s+1 <<"/"<<step<<endl;
    }
}

double perceptron::test()
{
    int m = X_test.size(), n = X_test[0].size();
    std::cout<<"Test start!"<<endl;
    double errCnt = 0;
    for (int i = 0; i < m; ++i)
    {
        vector<double> xi = X_test[i];
        double yi = y_test[i];
        double dotVal = 0.0;
        for (int j = 0; j < n; ++j)
        {
            dotVal += (w[j]*xi[j]);
        }
        if ((yi*(dotVal+b)) <= 0) errCnt++;
    }
    double acc = 1.0-errCnt/m;
    cout<<"Accuary rate ="<<acc<<endl;
    return acc;
}