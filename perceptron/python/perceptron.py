# encoding = utf-8
# author : renlei
'''
lr: 0.01
train:test = 8:2
iters: 30
Acc: 0.665
'''
import numpy as np
import time
def load_mnist(txt_data):
    f = open(txt_data, 'r')
    fr = f.readlines()
    dataArr, labelArr = [], []
    for line in fr:
        feats = line.strip().split(',')
        dataArr.append(feats[:-1])
        # mnist数据集标签有十种，即0,1,2,...,9.这里为了转换为二分类问题对标签重新处理
        # 其中0,1,2,3,4为正类. 5-9为负类
        labelArr.append( 1 if int(feats[-1]) >= 5 else -1)
    f.close()
    return np.array(dataArr, dtype=np.float), np.array(labelArr,dtype=np.int)

def train_test_split(x_data, y_data, train_rate):
    Idx = np.arange(len(x_data))
    np.random.shuffle(Idx) # shuffle in-place
    trainIdx, testIdx = Idx[:round(train_rate*len(x_data))], Idx[round(train_rate*len(x_data)):]
    x_train, y_train = x_data[trainIdx], y_data[trainIdx]
    x_test, y_test = x_data[testIdx], y_data[testIdx]
    return x_train, y_train, x_test, y_test
class Perceptron(object):
    def __init__(self):
        return 
    # 感知机算法的终止条件是训练集中没有误分类点
    # 这里为了方便计算，设置最大迭代次数终止训练  
    def train(self, x_tr, y_tr, lr=0.5, iters=50):
        m, n = x_tr.shape
        self.w = np.zeros((n,))
        self.b = 0
        for it in range(iters):
            for i in range(m):
                xi, yi = x_tr[i], y_tr[i]

                if (yi*np.dot(xi,self.w)+self.b) <= 0:
                    self.w, self.b = self.w+lr*yi*xi, self.b+lr*yi
            print("round {0}/{1}".format(it, iters))
        return self.w, self.b
    # 训练和测试中对误分类点的判断均为 (yi*np.dot(xi,self.w)+self.b) <= 0
    # 注意这里不能退化为 < .因为w和b的初始化为0值，这是满足等于0的条件的.
    # 造成的结果是 wx+b 恒等于 0
    def test(self, x_te, y_te):
        m, _ = x_te.shape
        errCnt = 0
        for i in range(m):
            xi, yi = x_te[i], y_te[i]
            if (yi*np.dot(xi,self.w)+self.b) <= 0:
                errCnt += 1
        accuary_rate = 1-errCnt/m
        print("Test accuary rate:{}".format(accuary_rate))
        return accuary_rate
if __name__=="__main__":
    start = time.time()
    datas, labels = load_mnist('_Data/mnist.txt')
    X_train, y_train, X_test, y_test = train_test_split(datas, labels, 0.8)
    print("Io cost time = {} s".format(time.time()-start))
    per = Perceptron() 
    per.train(X_train, y_train, 1e-2, 30)
    per.test(X_test, y_test)
    end = time.time()
    print("cost time = {} s".format(end-start))