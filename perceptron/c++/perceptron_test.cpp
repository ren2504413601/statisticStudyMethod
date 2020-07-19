#include"perceptron.h"
#include<iostream>
#include<ctime>
// lr: 0.01
// iters: 30
// accuary:0.82
// cost time: 21s
int main(int argc, char *argv[])
{
    vector<vector<double>> dataArr;
    string mnist_path = "_Data/mnist.txt";
    // std::clock() 返回进程从关联到程序执行的实现定义时期开始，
    // 所用的粗略处理器时间。为转换结果为秒，可将它除以 CLOCKS_PER_SEC 
    clock_t start=double(std::clock())/CLOCKS_PER_SEC ;
    perceptron per;
    dataArr = per.load_mnist(mnist_path);
    cout<<"IO cost time =" <<double(std::clock())/CLOCKS_PER_SEC-start<<"s."<<endl;
    per.splitData(dataArr, 0.8);

    per.train(30, 0.01);
    per.test();
    clock_t end=double(std::clock())/CLOCKS_PER_SEC ;
    cout<<"cost time ="<<end-start<<"s."<<endl;
    system("pause");
    return 0;
}