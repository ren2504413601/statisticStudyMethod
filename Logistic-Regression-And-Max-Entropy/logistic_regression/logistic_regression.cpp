#include<algorithm>
#include<sstream>
#include<fstream> 
#include<cmath>
#include"logistic_regression.h"

typedef unsigned long ui;

Logistic_regression::Logistic_regression()
{} 

vector<vector<double>> Logistic_regression::load_mnist(const string& file_name) // const引用（常量引用）
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
void Logistic_regression::splitData(vector<vector<double>>& dataArr, const double& train_rate)
{
    // random_shuffle 定义于头文件 <algorithm>
    std::random_shuffle(dataArr.begin(), dataArr.end());
    ui size = dataArr.size();
    ui trainSize = size * train_rate;
    std::cout<<"totoal data size ="<<size<<", training data size ="<< trainSize<<endl;
    for (int i = 0; i < trainSize; ++i)
    {
        vector<double> tmpData(dataArr[i]);
        if (tmpData.back() >= 5) y_train.push_back(1);
        else y_train.push_back(0);
        tmpData.pop_back();
        X_train.push_back(tmpData);
    }
    for (int j = trainSize; j < size; ++j)
    {
        vector<double> tmpData(dataArr[j]);
        if (tmpData.back() >= 5) y_test.push_back(1);
        else y_test.push_back(0);
        tmpData.pop_back();
        X_test.push_back(tmpData);
    }
}
/**
 * P(Y=1|x)=exp(wx+b)/(1+exp(wx+b)) P(Y=0|x)=1/(1+exp(wx+b))
 * 似然函数L(w)=\sum_{i=1}^N [yi(w xi+b)-log(1+exp(wx+b))]
 */
void Logistic_regression::train(const int& step,const float& lr)
{
    int m = X_train.size(), n = X_train[0].size();
    double sumY = 0.0;
    vector<double>sumXY(n, 0.0);
    w = vector<double>(n, 0);
    b = 0;
    std::cout<<"Training start!"<<endl;
    for (int i = 0; i < m; ++i)
    {
        sumY += y_train[i];
        for (int j = 0; j < n; ++j)
        {
            sumXY[j] += y_train[i]*X_train[i][j]; 
        }
    }
    for (int s = 0; s < step; ++s)
    {
        for (int i = 0; i < m; ++i)
        {
            double dotVal = 0.0;
            vector<double> xi = X_train[i];
            double yi = y_train[i];
            for (int j = 0; j < n; ++j) dotVal += (w[j]*X_train[i][j]);
            for (int j = 0; j < n; ++j)
            {
                double Wprime = yi*xi[j]-xi[j]*std::exp(dotVal+b)/(1+std::exp(dotVal+b));
                w[j] += lr*Wprime;
            }
            double Bprime = yi-std::exp(dotVal+b)/(1+std::exp(dotVal+b));
            b += lr*Bprime;
        }
        std::cout<<"round: "<< s+1 <<"/"<<step<<endl;
    }
}

int Logistic_regression::predict(vector<double>& x_test)
{
    double dotVal = 0.0;
    for (int i = 0; i < x_test.size(); ++i) dotVal += w[i]*x_test[i];
    if (1./(1+std::exp(dotVal+b)) >= 0.5) return 0;
    else return 1;
}

double Logistic_regression::test()
{
    int m = X_test.size(), n = X_test[0].size();
    std::cout<<"Test start!"<<endl;
    double errCnt = 0;
    for (int i = 0; i < m; ++i)
    {
        vector<double> xi = X_test[i];
        double yi = y_test[i];
        if (predict(xi) != yi) errCnt++;
    }
    double acc = 1.0-errCnt/m;
    cout<<"Accuary rate ="<<acc<<endl;
    return acc;
}